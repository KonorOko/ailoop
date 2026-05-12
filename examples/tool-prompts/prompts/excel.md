# Excel quirks

Specific guidance for the `excel_*` family on top of the shared
file-editing rules:

- Cell addresses use A1 notation. Ranges look like `A1:C10`.
- When writing a formula, escape the leading `=` so the tool stores
  it as a formula rather than a literal string.
- Numeric cells are returned as floats. Cast on the client side if
  you need integers.
