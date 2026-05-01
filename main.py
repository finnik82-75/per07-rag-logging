"""
Главный файл для запуска RAG-ассистента.

Это точка входа в приложение. Здесь происходит:
1. Загрузка конфигурации
2. Инициализация всех компонентов (кеш, векторная база, RAG)
3. Добавление примеров документов (при первом запуске)
4. Интерактивный цикл общения с пользователем
"""

import os
import time
from dotenv import load_dotenv
from embeddings import EmbeddingStore, get_sample_documents
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger


def initialize_system():
    """
    Инициализирует все компоненты RAG-системы.
    
    Returns:
        Кортеж (embedding_store, rag_assistant, cache)
    """
    print("=" * 70)
    print("🚀 ИНИЦИАЛИЗАЦИЯ RAG-АССИСТЕНТА")
    print("=" * 70)
    
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Проверяем наличие API ключа OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  ВНИМАНИЕ: Не найден OPENAI_API_KEY в переменных окружения!")
        print("   Создайте файл .env и добавьте туда: OPENAI_API_KEY=your_key_here")
        print("   Или установите переменную окружения в системе.")
        print()
    
    # 1. Инициализируем кеш для хранения ответов
    print("\n[1/3] Инициализация кеша...")
    cache = ResponseCache(cache_file="cache.json")
    
    # 2. Инициализируем векторное хранилище ChromaDB
    print("\n[2/3] Инициализация векторного хранилища...")
    embedding_store = EmbeddingStore(
        collection_name="rag_documents",
        persist_directory="./chroma_db",
        embedding_model="text-embedding-3-small",  # Модель OpenAI для эмбеддингов
        api_key=api_key
    )
    
    # Проверяем, нужно ли добавить примеры документов
    if embedding_store.collection.count() == 0:
        print("\n📝 База данных пуста. Добавляем примеры документов...")
        sample_docs = get_sample_documents()
        embedding_store.add_documents(sample_docs)
    else:
        print(f"✓ В базе уже есть {embedding_store.collection.count()} документов")
    
    # 3. Инициализируем RAG-ассистента
    print("\n[3/3] Инициализация RAG-ассистента...")
    rag_assistant = RAGAssistant(
        embedding_store=embedding_store,
        model="gpt-3.5-turbo",
        temperature=0.7
    )

    logger = DatabaseLogger()
    
    print("\n" + "=" * 70)
    print("✅ СИСТЕМА ГОТОВА К РАБОТЕ")
    print("=" * 70)
    
    return embedding_store, rag_assistant, cache, logger


def answer_question(
    query: str,
    rag_assistant: RAGAssistant,
    cache: ResponseCache,
    logger: DatabaseLogger,
    source: str = "console",
) -> str:
    """
    Отвечает на вопрос пользователя с использованием кеша и RAG.
    
    Логика работы:
    1. Проверяем кеш - если ответ есть, возвращаем его
    2. Если ответа нет, выполняем RAG (поиск + генерация)
    3. Сохраняем новый ответ в кеш
    4. Возвращаем ответ
    
    Args:
        query: Вопрос пользователя
        rag_assistant: Экземпляр RAG-ассистента
        cache: Экземпляр кеша
        
    Returns:
        Ответ на вопрос
    """
    print("\n" + "=" * 70)
    print(f"❓ ВОПРОС: {query}")
    print("=" * 70)

    start_time = time.time()
    answer = None
    from_cache = False
    
    # Шаг 1: Проверяем кеш
    print("\n[Шаг 1] Проверка кеша...")
    cached_answer = cache.get(query)
    
    if cached_answer:
        answer = cached_answer
        from_cache = True
        # Ответ найден в кеше
        print("\n💾 Ответ из кеша:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
    else:
        # Шаг 2: Ответа нет в кеше - выполняем RAG
        print("\n[Шаг 2] Выполнение RAG (поиск + генерация)...")

        try:
            answer, search_results = rag_assistant.generate_response(
                query=query,
                top_k=7,
                verbose=True
            )

            # Шаг 3: Сохраняем ответ в кеш
            print("\n[Шаг 3] Сохранение ответа в кеш...")
            cache.set(query, answer)

            # Выводим финальный ответ
            print("\n💡 ОТВЕТ:")
            print("-" * 70)
            print(answer)
            print("-" * 70)

        except Exception as e:
            answer = f"Ошибка при обработке запроса: {str(e)}"
            print(f"\n❌ {answer}")
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.log_interaction(
                query=query,
                response=answer,
                source=source,
                user_id="local",
                username="local_user",
                from_cache=from_cache,
                response_time_ms=response_time_ms,
            )
            return answer

    response_time_ms = int((time.time() - start_time) * 1000)
    logger.log_interaction(
        query=query,
        response=answer,
        source=source,
        user_id="local",
        username="local_user",
        from_cache=from_cache,
        response_time_ms=response_time_ms,
    )
    return answer


def interactive_mode(rag_assistant: RAGAssistant, cache: ResponseCache, logger: DatabaseLogger):
    """
    Интерактивный режим общения с ассистентом.
    
    Пользователь может задавать вопросы в цикле до тех пор,
    пока не введет команду выхода.
    """
    print("\n" + "=" * 70)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nВы можете задавать вопросы ассистенту.")
    print("Для выхода введите: exit, quit, выход или q")
    print("\nДоступные команды:")
    print("  • cache - показать информацию о кеше")
    print("  • clear_cache - очистить кеш")
    print("  • stats - показать статистику системы")
    print()
    
    while True:
        try:
            # Получаем ввод от пользователя
            user_input = input("\n👤 Вы: ").strip()
            
            # Проверяем команды выхода
            if user_input.lower() in ['exit', 'quit', 'выход', 'q', '']:
                print("\n👋 До свидания!")
                break
            
            # Обрабатываем специальные команды
            if user_input.lower() == 'cache':
                response = f"📊 Кеш содержит {cache.size()} записей"
                print(f"\n{response}")
                logger.log_interaction(
                    query=user_input,
                    response=response,
                    source="interactive_command",
                    user_id="local",
                    username="local_user",
                    from_cache=False,
                    response_time_ms=0,
                )
                continue
            
            if user_input.lower() == 'clear_cache':
                cache.clear()
                response = "✓ Кеш очищен"
                print(f"\n{response}")
                logger.log_interaction(
                    query=user_input,
                    response=response,
                    source="interactive_command",
                    user_id="local",
                    username="local_user",
                    from_cache=False,
                    response_time_ms=0,
                )
                continue
            
            if user_input.lower() == 'stats':
                db_stats = logger.get_stats()
                response_lines = [
                    "📊 СТАТИСТИКА СИСТЕМЫ:",
                    f"  • Документов в ChromaDB: {rag_assistant.embedding_store.collection.count()}",
                    f"  • Записей в кеше: {cache.size()}",
                    f"  • Модель LLM: {rag_assistant.model}",
                    f"  • Всего запросов: {db_stats['total_requests']}",
                    f"  • Попаданий в кеш: {db_stats['cache_hits']}",
                    f"  • Среднее время ответа (мс): {db_stats['avg_response_time_ms']:.2f}",
                ]
                response = "\n".join(response_lines)
                print(f"\n{response}")
                logger.log_interaction(
                    query=user_input,
                    response=response,
                    source="interactive_command",
                    user_id="local",
                    username="local_user",
                    from_cache=False,
                    response_time_ms=0,
                )
                continue
            
            # Обрабатываем вопрос пользователя
            answer_question(user_input, rag_assistant, cache, logger, source="interactive")
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")


def demo_mode(rag_assistant: RAGAssistant, cache: ResponseCache, logger: DatabaseLogger):
    """
    Демонстрационный режим с заранее заготовленными вопросами.
    
    Показывает работу системы на примерах, включая использование кеша.
    """
    print("\n" + "=" * 70)
    print("🎬 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nСейчас будет продемонстрирована работа RAG-ассистента")
    print("на нескольких примерах вопросов.\n")
    
    # Список демо-вопросов
    demo_questions = [
        "Что такое Python и для чего он используется?",
        "Расскажи про RAG и как он работает",
        "Что такое векторные базы данных?",
        "Что такое Python и для чего он используется?"  # Повторный вопрос для демонстрации кеша
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'#' * 70}")
        print(f"ВОПРОС {i} из {len(demo_questions)}")
        print(f"{'#' * 70}")
        
        answer_question(question, rag_assistant, cache, logger, source="demo")
        
        # Пауза между вопросами (кроме последнего)
        if i < len(demo_questions):
            input("\n[Нажмите Enter для следующего вопроса...]")
    
    print("\n\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)


def main():
    """
    Главная функция приложения.
    """
    try:
        # Инициализируем систему
        embedding_store, rag_assistant, cache, logger = initialize_system()
        
        # Выбор режима работы
        print("\n" + "=" * 70)
        print("ВЫБОР РЕЖИМА РАБОТЫ")
        print("=" * 70)
        print("\n1. Интерактивный режим - задавайте свои вопросы")
        print("2. Демонстрационный режим - готовые примеры вопросов")
        print()
        
        mode = input("Выберите режим (1 или 2, по умолчанию 1): ").strip()
        
        if mode == '2':
            demo_mode(rag_assistant, cache, logger)
            
            # Предложить перейти в интерактивный режим
            print("\n" + "=" * 70)
            continue_interactive = input("\nПерейти в интерактивный режим? (y/n): ").strip().lower()
            if continue_interactive in ['y', 'yes', 'д', 'да', '']:
                interactive_mode(rag_assistant, cache, logger)
        else:
            interactive_mode(rag_assistant, cache, logger)
        
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

