"use strict";

function process_text(text) {
    // numeric, ^ and $ are important here for exact match
    if (/^-?[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)$/.test(text)) {
        return text;
    }
    // href text processing
    if (/^<a (.|\n)*<\/a>$/.test(text)) {
        text = text.replace(/(<a [^>]*>)|(<\/a>)/g, "")
    }
    return '"' + text.trim().replace(/"/g, '""') + '"';
}

function export_to_csv(id, table, export_name) {
    let title = [];
    // Use this jquery API till DataTables 2.0 is out
    let i = 0;  // Columns counter
    let headerProcessed = false;
    $('#' + id).find('tr').each(function () {
        if (headerProcessed) {
            return;
        }
        $(this).find('th').each(function () {
            if (headerProcessed) {
                return;
            }
            // Ignore first index column
            if (i === 0) {
                i++;
                return;
            }
            let text = $(this).text();
            // jQuery DataTable inserts trailing columns 1, 2, 3, ...
            // ignore them
            if (text === "1") {
                headerProcessed = true;
                return;
            }
            title.push(text);
            i++;
        });
    });
    title = title.join(",")

    let rows = [];
    table.rows().every(function (rowIdx, tableLoop, rowLoop) {
        const data = table.row(rowIdx).data();
        const row = [];
        // Ignore first index column
        for (let c = 1; c < data.length; c++) {
            row.push(process_text(data[c].toString()));
        }
        rows.push(row.join(","));
    });
    const csv = title + "\n" + rows.join("\n");

    // Initiate download action
    const uri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
    const download_link = document.createElement('a');
    download_link.href = uri;
    download_link.download = export_name + ".csv";
    document.body.appendChild(download_link);
    download_link.click();
    document.body.removeChild(download_link);
}

