#!/usr/bin/env python3
"""
Daily Coach Report PDF Generator

Combines daily feedback and details markdown files into a single PDF.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import markdown
    from weasyprint import HTML
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install with: poetry add markdown weasyprint")
    sys.exit(1)


def find_daily_files(date_str: str, athlete: str, base_dir: str) -> tuple[str | None, str | None]:
    """Find the feedback and details files for a given date and athlete."""
    daily_dir = Path(base_dir) / "athletes" / athlete.lower() / "daily"
    
    if not daily_dir.exists():
        return None, None
    
    # Current naming pattern (without athlete name in filename)
    feedback_file = daily_dir / f"{date_str}_feedback.md"
    details_file = daily_dir / f"{date_str}_details.md"
    
    if feedback_file.exists() and details_file.exists():
        return str(feedback_file), str(details_file)
    
    # Fallback: Try older pattern with athlete name
    feedback_file = daily_dir / f"{date_str}_{athlete.capitalize()}_feedback.md"
    details_file = daily_dir / f"{date_str}_{athlete.capitalize()}_details.md"
    
    if feedback_file.exists() and details_file.exists():
        return str(feedback_file), str(details_file)
    
    return None, None


def read_markdown_file(file_path: str) -> str:
    """Read and return the content of a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def combine_markdown_content(feedback_content: str, details_content: str, date_str: str, athlete: str) -> str:
    """Combine feedback and details content into a single markdown document."""
    combined_content = f"""# Daily Coach Report - {athlete.capitalize()}
**Date:** {date_str}

---

{feedback_content.strip()}

---

{details_content.strip()}
"""
    return combined_content


def markdown_to_pdf(markdown_content: str, output_path: str) -> bool:
    """Convert markdown content to PDF using weasyprint."""
    try:
        # Convert markdown to HTML
        md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
        html_content = md.convert(markdown_content)
        
        # Add basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #7f8c8d;
                }}
                hr {{
                    border: none;
                    height: 1px;
                    background-color: #bdc3c7;
                    margin: 30px 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                code {{
                    background-color: #f8f8f8;
                    padding: 2px 4px;
                    border-radius: 3px;
                }}
                pre {{
                    background-color: #f8f8f8;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    margin-left: 0;
                    font-style: italic;
                    color: #7f8c8d;
                }}
                strong {{
                    color: #2c3e50;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        html_doc = HTML(string=styled_html)
        pdf_bytes = html_doc.write_pdf()
        
        # Check if PDF generation succeeded
        if pdf_bytes is None:
            raise Exception("PDF generation returned None")
            
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
        return True
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False


def validate_date(date_str: str) -> bool:
    """Validate that the date string is in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF from daily coaching output files"
    )
    
    parser.add_argument('date', help='Date in YYYY-MM-DD format')
    parser.add_argument('athlete', help='Athlete name')
    parser.add_argument('--exchange-dir', default='../ENV/exchange', 
                        help='Path to the exchange directory (default: ../ENV/exchange)')
    parser.add_argument('--output-dir', 
                        help='Output directory for PDF (default: same as input files)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_date(args.date):
        print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD format.")
        sys.exit(1)
    
    # Find the daily files
    feedback_path, details_path = find_daily_files(args.date, args.athlete, args.exchange_dir)
    
    if not feedback_path or not details_path:
        print(f"Error: Could not find both feedback and details files for {args.athlete} on {args.date}")
        print(f"Searched in: {Path(args.exchange_dir) / 'athletes' / args.athlete.lower() / 'daily'}")
        
        # List available files for debugging
        daily_dir = Path(args.exchange_dir) / "athletes" / args.athlete.lower() / "daily"
        if daily_dir.exists():
            files = list(daily_dir.glob(f"{args.date}*"))
            if files:
                print(f"Available files for {args.date}:")
                for file in files:
                    print(f"  {file.name}")
            else:
                print(f"No files found for date {args.date}")
        else:
            print(f"Directory does not exist: {daily_dir}")
        sys.exit(1)
    
    print(f"Found files:")
    print(f"  Feedback: {feedback_path}")
    print(f"  Details: {details_path}")
    
    # Read the markdown files
    feedback_content = read_markdown_file(feedback_path)
    details_content = read_markdown_file(details_path)
    
    if not feedback_content and not details_content:
        print("Error: Both files appear to be empty or unreadable.")
        sys.exit(1)
    
    # Combine the content
    combined_content = combine_markdown_content(feedback_content, details_content, args.date, args.athlete)
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use the same directory as the input files
        output_dir = Path(feedback_path).parent
    
    output_filename = f"{args.date}_{args.athlete}_daily_report.pdf"
    output_path = output_dir / output_filename
    
    # Generate PDF
    print(f"Generating PDF: {output_path}")
    if markdown_to_pdf(combined_content, str(output_path)):
        print(f"✅ PDF generated successfully: {output_path}")
    else:
        print("❌ Failed to generate PDF")
        sys.exit(1)


if __name__ == "__main__":
    main()