from googletrans import Translator

def translate(text):
    """
    Переводит текст с русского на английский язык.
    
    Args:
        text (str): Текст на русском языке для перевода
        
    Returns:
        str: Переведенный текст на английском языке
    """
    # Создаем экземпляр переводчика
    translator = Translator()
    
    # Выполняем перевод
    result = translator.translate(text, src='ru', dest='en')
    
    # Возвращаем переведенный текст
    return result.text

if __name__ == "__main__":
    # Пример использования
    test_text = input("Введите текст для перевода: ")
    translated = translate(test_text)
    print(f"Оригинал: {test_text}")
    print(f"Перевод: {translated}")
