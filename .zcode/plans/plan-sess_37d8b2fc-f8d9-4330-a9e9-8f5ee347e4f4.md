## LlamaIndex поверх существующей инфраструктуры (режим A/B)

**Идея:** не выбрасываем рукописный путь — он остаётся базовой линией сравнения и материалом для обучения. LlamaIndex добавляется как вторая реализация того же пайплайна поверх той же инфраструктуры (Postgres = источник правды, тот же сервер Qdrant, те же e5-векторы fastembed, тот же llama.cpp-LLM). На собесе это сильный рассказ: «умею и руками, и фреймворком; понимаю, что именно фреймворк делает под капотом».

### Архитектурные решения (и почему)

1. **Worker и Postgres-чанковка остаются владельцами данных.** LlamaIndex не пересоздаёт чанки (у его нод были бы другие id/границы) — синхронизация читает готовые чанки из Postgres и строит `TextNode` с `id_ = str(chunk.id)`, чтобы UUID связывал оба хранилища (тот же FK-трюк, что уже в `app/vectorstore.py`). В проде так и делают: БД — system of record, индекс — производное.
2. **Отдельная коллекция Qdrant `llama_doc_chunks`.** У `QdrantVectorStore` свой payload-схема (`text` + `metadata.*`); смешивать двух писателей в одной коллекции — каша. Две коллекции — честная картина «параллельные индексы при миграции/сравнении».
3. **Общая точка входа — embeddinг-адаптер.** Пишем свой `BaseEmbedding`-класс (`E5FastEmbed`), который внутри вызывает существующий `embedding_service` (асинхронные `_aget_*` — батчи через `embed_passages`). Это гарантирует *идентичные векторы* с рукописным путём → скоринги в A/B сравнимы, и это же типовой продакшен-паттерн «свой провайдер внутрь чужого фреймворка».
4. **Полный library-first запрос:** `RetrieverQueryEngine` (retriever + ResponseSynthesizer с нашей системной инструкцией «отвечай только по контексту, ссылайся на сервис», fallback как сейчас) — LlamaIndex используется целиком, а не только как обёртка поиска. Провенанс берём из `response.source_nodes`.

### Файлы

- `pyproject.toml`: + `llama-index-core`, + `llama-index-vector-stores-qdrant`, + `llama-index-llms-openai-like` (резолв проверен: core 0.14.25, qdrant-store 0.10.4, openai-like 0.8.1; риск transitive-wheel на 3.14 и пересечения версий `openai` SDK — проверяю сразу после `uv add`, откатной путь описан ниже).
- `app/config.py`: + `LLAMA_QDRANT_COLLECTION: str = "llama_doc_chunks"`.
- `app/llama_rag.py` (новый):
  - `E5FastEmbed(BaseEmbedding)` — адаптер к embedding_service.
  - `build_index()` — `QdrantVectorStore` поверх существующего сервера (client/аclient — как требует версия), `VectorStoreInfo` с metadata-полем `service_name` для фильтров, `Collection.from_existing`/`VectorStoreIndex.from_vector_store`.
  - `sync_from_postgres()` — full rebuild: читает чанки join services → TextNodes (id=UUID, metadata: service_name/service_id/chunk_index/doc_file_path) → truncate коллекции → `await pipeline.arun(...)` через `IngestionPipeline(transformations=[embeddings], vector_store=...)`.
  - `build_query_engine(top_k, service_filter)` — `OpenAILike(model=LLM_API_BASE-модель, api_base=LLM_API_BASE, api_key=LLM_API_KEY, is_chat_model=True)` + retriever с опциональным `MetadataFilters(service_name=...)` + синтез с нашей prompt-инструкцией.
  - CLI: `python -m app.llama_rag sync` (запускать после изменений корпуса, как сейчас `graph_neo4j ingest`).
- `main.py`: в `ChatRequest` поле `mode: Literal["handwritten","llamaindex"] = "handwritten"`; ветвление в `/chat`: handwritten — текущий путь; llamaindex — `build_query_engine().aquery(...)`, `reply = str(response)`, sources из `source_nodes` (score там тоже cosine). Fallback при недоступном LLM — вывод контекста, как сейчас.
- `tests/test_llama_rag.py`: детерминированный смоук **без LLM**: собрать индекс из 2 фейковых чанков во временной коллекции, embed → retrieve → проверить порядок и `MetadataFilters` по service_name; teardown — удалить коллекцию. Встраивается в текущий pytest-раннер (asyncio auto, session loop).

### Проверка (build → run → observe)

1. `uv add` + импорт всех модулей (первый же шаг вскрывает конфликты версий).
2. `python -m app.llama_rag sync` → сверить: Postgres-чанков 102 == точек в `llama_doc_chunks`; посмотреть в dashboard localhost:6333 (у llama-точек payload другой формы — наглядно сравнить с нашей ручной коллекцией).
3. Живой прогон: одинаковые вопросы («recurring payments», «JWT» + filter auth) через `/chat` в обоих режимах — сравнить sources и скоры (должны совпадать с точностью до ANN-рандомизации), reply от 1.5B модели.
4. `uv run --extra dev pytest` — всё зелёное.
5. Коммит.

### Риски / откатные пути

- Транзитивные wheels llama-index на Python 3.14 — почти наверняка ОК (все pure-python или с свежими wheels), но проверю первым шагом.
- Пересечение версий пакета `openai` между langchain-openai и llama-index-llms-openai — при конфликте фиксирую версию, в крайнем случае заменяю `OpenAILike` на тонкий свой LLM-адаптер к тому же эндпоинту (тот же паттерн, что с эмбеддингами).
- Качество ответов 1.5B через дефолтные промпты LlamaIndex хуже нашего — лечится кастомным `ResponseSynthesizer`-шаблоном, входящим в план.

### Что сознательно НЕ делаем сейчас

- Не подключаем PropertyGraphIndex/graph-стор LlamaIndex к Neo4j — следующий этап, после уроков Cypher: наш `graph_neo4j.expand` тогда встроим в A/B как **кастомный retriever** внутри LlamaIndex (и это же даст «третью ветку» сравнения).
- Не трогаем worker (llama-синхронизация отдельно, чтобы не сдваивать эмбеддинги каждые 60 с).