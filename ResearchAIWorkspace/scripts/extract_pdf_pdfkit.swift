import Foundation
import PDFKit

guard CommandLine.arguments.count == 2 else {
    fputs("Usage: extract_pdf_pdfkit <pdf-path>\n", stderr)
    exit(2)
}

let path = CommandLine.arguments[1]
let url = URL(fileURLWithPath: path)

guard let document = PDFDocument(url: url) else {
    fputs("Unable to open PDF: \(path)\n", stderr)
    exit(1)
}

print("<<<PDFKIT_PAGE_COUNT:\(document.pageCount)>>>")
for index in 0..<document.pageCount {
    print("<<<PDFKIT_PAGE:\(index + 1)>>>")
    print(document.page(at: index)?.string ?? "")
}
