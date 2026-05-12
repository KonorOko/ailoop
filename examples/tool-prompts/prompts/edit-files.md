# Editing files

You can edit Excel, Word and PDF files via dedicated tools. Common
rules across every editor:

- Always read the file first with the matching `*_read` tool (or
  `pdf_read` for PDFs) before writing — never edit blind.
- Quote a short snippet of the surrounding text so the user can
  verify the edit landed in the right place.
- If the file is missing, do not create a new one silently — confirm
  with the user that creation (not edit) is the intent.
