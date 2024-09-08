import os
import time
import tempfile
import re
from typing import List, Tuple
from pathlib import Path
from litellm import completion
from litellm.exceptions import RateLimitError

def extract_toc(text: str) -> List[str]:
    prompt = """
    Analyze the following text and identify the table of contents.
    Return ONLY a list of chapter names, one per line, in the order they appear, without any introductory text or numbering.
    Do not include the chapter number and do not preface the list with a statement like 'Here is list of chapter names'.
    If no table of contents is found, return an empty list.
    
    Example input:
    CONTENTS
    Dedication
    Prologue: A New Beginning
    1. The First Step
    2. Challenges Ahead
    3. Overcoming Obstacles
    4. The Final Push
    Epilogue: Looking Back
    Acknowledgements
    
    Example output:
    The First Step
    Challenges Ahead
    Overcoming Obstacles
    The Final Push
    
    Now, analyze this text:
    {text}
    """

    response = completion(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": prompt.format(text=text[:2000])}],
    )

    response_text = response.choices[0].message.content
    chapters = [line.strip() for line in response_text.split("\n") if line.strip()]
    return chapters

def find_chapter_occurrences(book_content: str, chapters: List[str]) -> List[Tuple[str, int]]:
    occurrences = []
    lines = book_content.split("\n")
    for i, line in enumerate(lines, 1):
        for chapter in chapters:
            if re.search(rf"^(\d+\.)?\s*{re.escape(chapter)}", line, re.IGNORECASE):
                occurrences.append((chapter, i))
    return occurrences

def merge_nearby_occurrences(occurrences: List[Tuple[str, int]], max_gap: int = 10) -> List[Tuple[str, int]]:
    merged = []
    for chapter, line_num in occurrences:
        if not merged or chapter != merged[-1][0] or line_num - merged[-1][1] > max_gap:
            merged.append((chapter, line_num))
    return merged

def filter_occurrences(occurrences: List[Tuple[str, int]], min_gap: int = 20) -> List[Tuple[str, int]]:
    filtered = []
    for i, (chapter, line_num) in enumerate(occurrences):
        if i == 0 or line_num - occurrences[i - 1][1] >= min_gap:
            filtered.append((chapter, line_num))
    return filtered

def split_book_into_chapters(book_content: str, chapter_starts: List[Tuple[str, int]]) -> dict:
    lines = book_content.split("\n")
    chapter_content = {}

    if chapter_starts[0][1] > 1:
        chapter_content["0000_Preface"] = "\n".join(lines[: chapter_starts[0][1] - 1]).strip()

    for i, (chapter, start_line) in enumerate(chapter_starts):
        end_line = chapter_starts[i + 1][1] if i < len(chapter_starts) - 1 else len(lines)
        content = "\n".join(lines[start_line - 1 : end_line]).strip()
        chapter_content[f"{start_line:04d}_{chapter}"] = content

    return chapter_content

def find_books(library_path: str) -> List[Tuple[str, str]]:
    books = []
    for root, dirs, files in os.walk(library_path):
        for file in files:
            if file.endswith(".txt"):
                author = os.path.basename(os.path.dirname(os.path.dirname(root)))
                title = os.path.splitext(file)[0]
                books.append((f"{author} - {title}", os.path.join(root, file)))
    return books

def analyze_chapter(chapter_content: str, chapter_name: str, scratchpad: str, model: str, max_retries: int = 5, initial_wait: float = 1.0) -> str:
    prompt = f"""
    Analyze the following chapter: {chapter_name}

    Chapter content:
    {chapter_content}

    Previous analysis (scratchpad):
    {scratchpad}

    Tasks:
    1. Write a comprehensive markdown summary of this chapter. Include key ideas, arguments, and any significant examples or case studies.
    2. Identify and list potential Zettelkasten notes (atomic ideas) from this chapter. Format each as a brief title followed by a concise explanation.
    3. List any authors, books, or papers mentioned in this chapter. If possible, provide full citations.
    4. If you identify a key topic (very relevant "proper nouns" such as an animal species, social movement, period of history or person of note) mentioned in the chapter. Create Wikipedia-style markdown links for these, e.g., [topic](https://en.wikipedia.org/wiki/Topic).
    5. List any open questions or thoughts for the next chapter.
    6. If this chapter appears to be a table of contents, appendix, or references, please note this and provide a brief description instead of a full analysis.
    7. Highlight any particularly insightful quotes from the chapter, using proper markdown formatting.

    Format your response in markdown, starting with a header for the chapter name, followed by your summary, Zettelkasten notes, references, key topics, questions, and other notes. Use appropriate markdown formatting for sections, lists, and links.

    Ensure your analysis is thorough but concise, focusing on the most important and interesting aspects of the chapter.
    """

    for attempt in range(max_retries):
        try:
            response = completion(model=model, messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. Unable to analyze chapter: {chapter_name}")
                return f"Error: Unable to analyze chapter due to rate limiting. Please try again later.\n\nChapter: {chapter_name}"
        except Exception as e:
            print(f"An error occurred while analyzing chapter {chapter_name}: {str(e)}")
            return f"Error: An unexpected error occurred while analyzing the chapter.\n\nChapter: {chapter_name}\nError: {str(e)}"

def select_model():
    models = ["claude-3-sonnet-20240229", "gpt-4", "ollama/llama3.1", "ollama/mistral-nemo"]
    print("Select a model:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(models):
                return models[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_book(library_path: str) -> str:
    books = find_books(library_path)
    print("\nSelect a book:")
    for i, (title, _) in enumerate(books, 1):
        print(f"{i}. {title}")
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(books):
                return books[choice - 1][1]  # Return the path of the selected book
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def answer_question(question: str, chapter_content: str, chapter_name: str, model: str) -> str:
    prompt = f"""
    You are analyzing the following chapter: {chapter_name}

    Chapter content:
    {chapter_content}

    The user has asked the following question about this chapter:
    {question}

    Please provide a clear and concise answer to the user's question, focusing specifically on the content of this chapter. Use markdown formatting for your response.
    """

    try:
        response = completion(model=model, messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: An error occurred while answering the question. Error: {str(e)}"

def get_safe_filename(filename: str) -> str:
    """
    Convert a string into a safe filename by removing or replacing invalid characters.
    """
    # Remove or replace invalid filename characters
    safe_filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    safe_filename = safe_filename.replace(' ', '_')
    # Limit filename length (optional, adjust as needed)
    safe_filename = safe_filename[:200]  # Limit to 200 characters
    return safe_filename

def main():
    model = select_model()
    print(f"Selected model: {model}")

    library_path = "/Users/wesleyflorence/Calibre Library"
    book_path = select_book(library_path)
    print(f"Selected book: {book_path}")

    # Extract book title from the path
    book_title = os.path.splitext(os.path.basename(book_path))[0]
    safe_book_title = get_safe_filename(book_title)

    with open(book_path, "r", encoding="utf-8") as file:
        book_content = file.read()

    chapters = extract_toc(book_content[:5000])
    if not chapters:
        print("No table of contents found. Unable to split the book.")
        return

    occurrences = find_chapter_occurrences(book_content, chapters)
    merged_occurrences = merge_nearby_occurrences(occurrences)
    filtered_occurrences = filter_occurrences(merged_occurrences)
    chapter_content = split_book_into_chapters(book_content, filtered_occurrences)

    temp_dir = tempfile.mkdtemp()
    desktop_path = os.path.expanduser("~/Desktop")
    analysis_file_path = os.path.join(desktop_path, f"{safe_book_title}-ai-review.md")

    total_chapters = len(chapter_content)
    scratchpad = ""

    with open(analysis_file_path, "w", encoding="utf-8") as analysis_file:
        # Write the book title at the beginning of the file
        analysis_file.write(f"# AI Review: {book_title}\n\n")

        for i, (chapter_name, content) in enumerate(chapter_content.items(), 1):
            print(f"\nAnalyzing chapter {i}/{total_chapters}: {chapter_name}")
            file_name = f"{chapter_name.replace(' ', '_')}.txt"
            chapter_file_path = os.path.join(temp_dir, file_name)
            with open(chapter_file_path, "w", encoding="utf-8") as file:
                file.write(content)

            analysis = analyze_chapter(content, chapter_name, scratchpad, model)
            print(analysis)
            
            analysis_file.write(f"\n\n{analysis}\n\n---\n")

            while True:
                user_input = input("Enter a question about this chapter, press Enter to continue to the next chapter, or type 'q' to quit: ")
                if user_input.lower() == 'q':
                    print("Exiting the analysis.")
                    return
                elif user_input.strip() == "":
                    break
                else:
                    answer = answer_question(user_input, content, chapter_name, model)
                    print("\nAnswer:")
                    print(answer)
                    analysis_file.write(f"\n\n## User Question\n\n{user_input}\n\n## Answer\n\n{answer}\n\n---\n")

            scratchpad = analysis[-1000:]

    print(f"\nAnalysis complete! Results saved in {analysis_file_path}")

if __name__ == "__main__":
    main()
