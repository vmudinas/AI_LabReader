import os
import ollama
import pandas as pd
import re
from pypdf import PdfReader
from datetime import datetime

PDF_FOLDER = "bloodwork_pdfs"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_date_from_filename(filename):
    """Extract the reporting date from the filename."""
    match = re.search(r"([a-zA-Z]+)(\d{8})(_\d+)?\.pdf", filename, re.IGNORECASE)
    if match:
        date_str = match.group(2)
        try:
            return datetime.strptime(date_str, "%m%d%Y").date()  # Corrected date format
        except ValueError:
            print(f"Invalid date format in filename: {filename}")
            return None
    return None

def extract_lab_results(text):
    """Extract all lab results dynamically from the text."""
    lab_results = {}
    lines = text.split("\n")
    for line in lines:
        match = re.match(r"([A-Za-z ]+):?\s*([\d.]+)\s*(\w+)?", line)
        if match:
            test_name = match.group(1).strip()
            value = match.group(2)
            unit = match.group(3) if match.group(3) else ""

            try:
                value = float(value) if value != "." else None
            except ValueError:
                value = None

            if value is not None:
                lab_results[test_name] = {"value": value, "unit": unit}
    return lab_results

def calculate_averages(all_lab_results):
    """Calculate averages for each test across all dates."""
    averages = {}
    for test_name in set(test for date_results in all_lab_results.values() for test in date_results):
        values = [
            date_results[test_name]["value"]
            for date_results in all_lab_results.values()
            if test_name in date_results
        ]
        if values:
            averages[test_name] = sum(values) / len(values)
    return averages

def get_medical_advice(all_lab_results, averages):
    """Generate a summary based on all bloodwork results and trends."""
    df = pd.DataFrame(all_lab_results).T
    prompt = f"""
    A patient has undergone multiple blood tests on different dates. Here are the results:
    {df.to_string()}

    Here are the averages for the tests across these dates:
    {averages}

    Provide a comprehensive medical assessment based on trends across these tests.
    Highlight any concerning values, trends, and recommendations.
    """
    response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

def interactive_mode(all_lab_results, averages):
    """Allow the user to ask questions about their lab results."""
    print("\nYou can now ask questions about your lab results. Type 'exit' to quit.")
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() == "exit":
            break

        prompt = f"""
        The patient has the following lab results:
        {pd.DataFrame(all_lab_results).T.to_string()}

        Here are the averages for the tests across these dates:
        {averages}

        Question: {user_input}
        Answer concisely and accurately.
        """

        response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])
        print("\nResponse:", response['message']['content'])

def main():
    all_lab_results = {}
    all_test_names = set()

    # Iterate over all PDFs in the specified folder
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            date = extract_date_from_filename(filename)  # Extract date from filename
            if date:
                text = extract_text_from_pdf(pdf_path)
                lab_results = extract_lab_results(text)

                if lab_results:
                    # Add the lab results to the dictionary, organized by date
                    if date not in all_lab_results:
                        all_lab_results[date] = {}
                    all_lab_results[date].update(lab_results)
                    all_test_names.update(lab_results.keys())
                    print(f"Processed: {filename} (Date: {date})")
                else:
                    print(f"No valid bloodwork data found in {filename}")
            else:
                print(f"Filename does not match expected format: {filename}")

    # If lab results exist, calculate averages, get medical advice, and enter interactive mode
    if all_lab_results:
        averages = calculate_averages(all_lab_results)
        advice = get_medical_advice(all_lab_results, averages)
        print("\n--- Summary Across All Tests ---")
        print(advice)

        interactive_mode(all_lab_results, averages)
    else:
        print("No valid lab results found.")

if __name__ == "__main__":
    main()
