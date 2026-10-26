#!/usr/bin/env bash
# Cold filesystem backup/restore of the Neo4j /data directory.
# Community edition has no online backup, so Neo4j is stopped for the seconds
# it takes to snapshot /data; PostgreSQL and Qdrant are never touched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$REPO_ROOT/backups"

NEO4J_SERVICE="${NEO4J_SERVICE:-neo4j}"
BOLT_HOST="${BOLT_HOST:-127.0.0.1}"
BOLT_PORT="${BOLT_PORT:-7687}"
BOLT_TIMEOUT_SECONDS="${BOLT_TIMEOUT_SECONDS:-60}"

DATA_TYPE=""
DATA_SOURCE=""
CONTAINER_ID=""

compose() { docker compose --project-directory "$REPO_ROOT" "$@"; }

# compose ps -q hides stopped containers (compose >= 2.20 default), so the id
# is captured while the container is up and the --all form is the fallback.
container_id() {
    if [[ -z "$CONTAINER_ID" ]]; then
        CONTAINER_ID="$(compose ps -q "$NEO4J_SERVICE" || true)"
    fi
    if [[ -z "$CONTAINER_ID" ]]; then
        CONTAINER_ID="$(compose ps --all -q "$NEO4J_SERVICE" || true)"
    fi
    if [[ -z "$CONTAINER_ID" ]]; then
        echo "ERROR: cannot resolve container of service '$NEO4J_SERVICE'" >&2
        exit 1
    fi
    echo "$CONTAINER_ID"
}

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
  save [--name SUFFIX]
      Stop Neo4j, archive the contents of its /data directory to
      $BACKUP_DIR/neo4j_<YYYYmmdd_HHMMSS><-SUFFIX>.tar.gz,
      then restart Neo4j and wait until it answers on Bolt.

  restore <archive.tar.gz>
      DESTRUCTIVE: stop Neo4j, empty the current /data contents, extract the
      archive in their place, then restart Neo4j and wait for Bolt.

  status
      Print the resolved /data mount and the latest backup file.

PostgreSQL and Qdrant are not stopped by any command.
EOF
}

resolve_data_mounts() {
    # The /data mount of the neo4j container (works whether it is running or
    # stopped). Caches CONTAINER_ID for the later --volumes-from steps.
    local spec
    spec="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Type}} {{.Source}}{{end}}{{end}}' "$(container_id)")"
    if [[ -z "$spec" ]]; then
        echo "ERROR: no /data mount found on container of service '$NEO4J_SERVICE'" >&2
        exit 1
    fi
    DATA_TYPE="${spec%% *}"
    DATA_SOURCE="${spec#* }"
}

container_image() { docker inspect -f '{{.Config.Image}}' "$(container_id)"; }

# Bind mounts and native-Linux volume roots are real host directories; Docker
# Desktop keeps named volumes inside its VM where the inspect Source does not
# exist on the host, so those go through a throwaway container instead.
data_reachable_from_host() { [[ "$DATA_TYPE" == "bind" || -d "$DATA_SOURCE" ]]; }

create_archive() {
    local archive="$1"
    if data_reachable_from_host; then
        tar -czf "$archive" -C "$DATA_SOURCE" .
    else
        docker run --rm --volumes-from "$(container_id)" \
            --entrypoint sh "$(container_image)" -c 'tar -czf - -C /data .' > "$archive"
    fi
}

empty_data_dir() {
    if data_reachable_from_host; then
        find "$DATA_SOURCE" -mindepth 1 -delete
    else
        docker run --rm --volumes-from "$(container_id)" \
            --entrypoint sh "$(container_image)" -c 'find /data -mindepth 1 -exec rm -rf -- {} +'
    fi
}

extract_archive() {
    local archive="$1"
    if data_reachable_from_host; then
        tar -xzf "$archive" -C "$DATA_SOURCE"
    else
        docker run --rm -i --volumes-from "$(container_id)" \
            --entrypoint sh "$(container_image)" -c 'tar -xzf - -C /data' < "$archive"
    fi
}

wait_for_bolt() {
    local deadline=$((SECONDS + BOLT_TIMEOUT_SECONDS))
    echo "==> Waiting for Neo4j Bolt on ${BOLT_HOST}:${BOLT_PORT} (up to ${BOLT_TIMEOUT_SECONDS}s)"
    while ((SECONDS < deadline)); do
        if (exec 3<>"/dev/tcp/${BOLT_HOST}/${BOLT_PORT}") 2>/dev/null; then
            exec 3>&- 3<&-
            echo "==> Neo4j Bolt is answering."
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Neo4j did not answer on Bolt within ${BOLT_TIMEOUT_SECONDS}s" >&2
    return 1
}

# EXIT trap: the container must never be left stopped, even if a middle step
# fails. compose start is idempotent; the running check only keeps output clean.
ensure_neo4j_up() {
    if [[ -z "$(compose ps --status running -q "$NEO4J_SERVICE")" ]]; then
        echo "==> Restarting Neo4j: docker compose start $NEO4J_SERVICE"
        compose start "$NEO4J_SERVICE"
    fi
    wait_for_bolt
}

cmd_save() {
    local suffix=""
    if [[ "${1:-}" == "--name" ]]; then
        if [[ -z "${2:-}" ]]; then
            echo "ERROR: --name requires a value" >&2
            exit 1
        fi
        suffix="-$2"
    fi

    local timestamp archive
    timestamp="$(date +%Y%m%d_%H%M%S)"
    archive="$BACKUP_DIR/neo4j_${timestamp}${suffix}.tar.gz"
    mkdir -p "$BACKUP_DIR"

    echo "==> Resolving Neo4j /data mount"
    resolve_data_mounts
    echo "==> /data mount: type=$DATA_TYPE source=$DATA_SOURCE"

    trap ensure_neo4j_up EXIT

    echo "==> Stopping Neo4j only (PostgreSQL and Qdrant stay up)"
    compose stop "$NEO4J_SERVICE"

    echo "==> Archiving /data contents to $archive"
    create_archive "$archive"

    echo "==> Restarting Neo4j"
    compose start "$NEO4J_SERVICE"
    wait_for_bolt

    echo "==> Backup written: $archive"
    ls -lh "$archive" | awk '{print "==> size: " $5}'
    echo "==> Archive contents (first entries):"
    tar -tzf "$archive" | head
}

cmd_restore() {
    local archive="${1:-}"
    if [[ -z "$archive" ]]; then
        echo "ERROR: restore requires an archive argument" >&2
        usage >&2
        exit 1
    fi
    if [[ ! -f "$archive" ]]; then
        echo "ERROR: archive not found: $archive" >&2
        exit 1
    fi
    if [[ "$archive" != *.tar.gz ]]; then
        echo "ERROR: archive must be a .tar.gz file: $archive" >&2
        exit 1
    fi

    echo "==> Resolving Neo4j /data mount"
    resolve_data_mounts

    cat <<WARNING
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! DESTRUCTIVE OPERATION: the current contents of the Neo4j data dir
!! will be deleted and replaced by the archive:
!!   $archive
!! Data dir: type=$DATA_TYPE source=$DATA_SOURCE
!! Only Neo4j is stopped; PostgreSQL and Qdrant stay up.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING

    trap ensure_neo4j_up EXIT

    echo "==> Stopping Neo4j only"
    compose stop "$NEO4J_SERVICE"

    echo "==> Emptying the /data contents"
    empty_data_dir

    echo "==> Extracting $archive"
    extract_archive "$archive"

    echo "==> Restarting Neo4j"
    compose start "$NEO4J_SERVICE"
    wait_for_bolt

    echo "==> Restore complete: graph data replaced, Neo4j answers on Bolt."
}

cmd_status() {
    resolve_data_mounts
    echo "Neo4j /data mount: type=$DATA_TYPE source=$DATA_SOURCE"
    local latest
    latest="$(ls -1t "$BACKUP_DIR"/neo4j_*.tar.gz 2>/dev/null | head -n 1 || true)"
    if [[ -n "$latest" ]]; then
        echo "Latest backup: $latest"
        ls -lh "$latest" | awk '{print "size: " $5}'
    else
        echo "Latest backup: none (no neo4j_*.tar.gz in $BACKUP_DIR)"
    fi
    echo "Neo4j service state:"
    compose ps "$NEO4J_SERVICE"
}

main() {
    local cmd="${1:-}"
    case "$cmd" in
        save) shift; cmd_save "$@" ;;
        restore) [[ $# -eq 2 ]] || { usage >&2; exit 1; }; cmd_restore "$2" ;;
        status) cmd_status ;;
        help|-h|--help) usage ;;
        *) usage >&2; exit 1 ;;
    esac
}

main "$@"