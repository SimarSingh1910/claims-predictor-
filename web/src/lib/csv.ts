// Minimal client-side CSV parsing, used ONLY for the pre-upload preview and
// mandatory-field check. The authoritative parse + validation happens server-side
// in /api/score; this is a fast local courtesy so the user sees their data and any
// obvious problems before they submit.
//
// Handles quoted fields with embedded commas and doubled-quote escapes. It does
// NOT try to handle embedded newlines inside quotes (AHC exports don't have them),
// so row counting can split on line breaks safely.

export interface CsvPreview {
  columns: string[];
  rows: string[][]; // up to maxRows data rows, aligned to columns
  rowCount: number; // total data rows (excludes header)
}

function splitLine(line: string): string[] {
  const out: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      out.push(field);
      field = "";
    } else {
      field += c;
    }
  }
  out.push(field);
  return out.map((s) => s.trim());
}

export function parseCsvPreview(text: string, maxRows = 5): CsvPreview {
  // Normalise line endings, drop a trailing blank line if present.
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  while (lines.length && lines[lines.length - 1] === "") lines.pop();

  if (lines.length === 0) return { columns: [], rows: [], rowCount: 0 };

  const columns = splitLine(lines[0]);
  const dataLines = lines.slice(1);
  const rows = dataLines.slice(0, maxRows).map(splitLine);
  return { columns, rows, rowCount: dataLines.length };
}
