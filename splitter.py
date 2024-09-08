import re
import os
import time
from typing import List, Tuple
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

    # Extract the actual response text from the API response
    response_text = response.choices[0].message.content

    # Parse the response to extract chapter names
    chapters = [line.strip() for line in response_text.split("\n") if line.strip()]

    return chapters


def find_chapter_occurrences(
    book_content: str, chapters: List[str]
) -> List[Tuple[str, int]]:
    occurrences = []
    lines = book_content.split("\n")
    for i, line in enumerate(lines, 1):
        for chapter in chapters:
            # More precise regex: look for chapter at start of line, possibly preceded by a number
            if re.search(rf"^(\d+\.)?\s*{re.escape(chapter)}", line, re.IGNORECASE):
                occurrences.append((chapter, i))
    return occurrences


def merge_nearby_occurrences(
    occurrences: List[Tuple[str, int]], max_gap: int = 10
) -> List[Tuple[str, int]]:
    merged = []
    for chapter, line_num in occurrences:
        if not merged or chapter != merged[-1][0] or line_num - merged[-1][1] > max_gap:
            merged.append((chapter, line_num))
    return merged


def filter_occurrences(
    occurrences: List[Tuple[str, int]], min_gap: int = 20
) -> List[Tuple[str, int]]:
    filtered = []
    for i, (chapter, line_num) in enumerate(occurrences):
        if i == 0 or line_num - occurrences[i - 1][1] >= min_gap:
            filtered.append((chapter, line_num))
    return filtered


def split_book_into_chapters(
    book_content: str, chapter_starts: List[Tuple[str, int]]
) -> dict:
    lines = book_content.split("\n")
    chapter_content = {}

    # Add content before the first chapter
    if chapter_starts[0][1] > 1:
        chapter_content["0000_Preface"] = "\n".join(
            lines[: chapter_starts[0][1] - 1]
        ).strip()

    for i, (chapter, start_line) in enumerate(chapter_starts):
        end_line = (
            chapter_starts[i + 1][1] if i < len(chapter_starts) - 1 else len(lines)
        )
        content = "\n".join(lines[start_line - 1 : end_line]).strip()
        chapter_content[f"{start_line:04d}_{chapter}"] = content

    return chapter_content


def analyze_chapter(chapter_content: str, chapter_name: str, scratchpad: str, max_retries: int = 5, initial_wait: float = 1.0) -> str:
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
            response = completion(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                print(f"Rate limit reached. Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached. Unable to analyze chapter: {chapter_name}")
                return f"Error: Unable to analyze chapter due to rate limiting. Please try again later.\n\nChapter: {chapter_name}"
        except Exception as e:
            print(f"An error occurred while analyzing chapter {chapter_name}: {str(e)}")
            return f"Error: An unexpected error occurred while analyzing the chapter.\n\nChapter: {chapter_name}\nError: {str(e)}"

# The rest of the script remains the same

# The rest of the script remains the same
def main(book_file_path: str):
    with open(book_file_path, "r", encoding="utf-8") as file:
        book_content = file.read()

    chapters = extract_toc(book_content[:5000])

    if not chapters:
        print("No table of contents found. Unable to split the book.")
        return

    print(f"Chapters found: {chapters}")

    occurrences = find_chapter_occurrences(book_content, chapters)
    merged_occurrences = merge_nearby_occurrences(occurrences)
    filtered_occurrences = filter_occurrences(merged_occurrences)
    chapter_content = split_book_into_chapters(book_content, filtered_occurrences)

    os.makedirs("chapters", exist_ok=True)

    scratchpad = ""
    analysis_file_path = "book_analysis.md"

    with open(analysis_file_path, "w", encoding="utf-8") as analysis_file:
        for chapter_name, content in chapter_content.items():
            file_name = f"{chapter_name.replace(' ', '_')}.txt"
            with open(f"chapters/{file_name}", "w", encoding="utf-8") as file:
                file.write(content)

            # Analyze chapter with Claude
            analysis = analyze_chapter(content, chapter_name, scratchpad)

            # Append analysis to the markdown file
            analysis_file.write(f"\n\n{analysis}\n\n---\n")
            time.sleep(1)

            # Update scratchpad (use the last 1000 characters to keep it manageable)
            scratchpad = analysis[-1000:]

            print(f"Completed analysis for {chapter_name}")

    print(
        f"Book split into {len(chapter_content)} chapters and analyzed. Results saved in {analysis_file_path}"
    )


if __name__ == "__main__":
    book_file_path = "multitudes.txt"
    main(book_file_path)
