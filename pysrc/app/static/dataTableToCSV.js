/*
Script to save table element postprocessed by jQuery DataTable plugin.
Based on https://www.jqueryscript.net/table/jQuery-Plugin-To-Convert-HTML-Table-To-CSV-tabletoCSV.html
*/
"use strict";

function process_text(text) {
    if (/-?[0-9]+(\.[0-9]+)?([eE]-?[0-9]+)/.test(text)) {
        return text;
    }
    return '"' + text.trim().replace(/"/g, '""') + '"';
}

function dataTableToCSV($table, name) {
    let title = [];
    let rows = [];

    let i = 0;  // Real columns counter
    let trailing = false;
    $table.find('tr').each(function () {
        let data = [];
        // jQuery DataTable inserts trailing columns 1, 2, 3, ...
        if (!trailing) {
            $(this).find('th').each(function () {
                // Ignore trailing columns
                if (trailing) {
                    return;
                }
                // Ignore first index
                if (i === 0) {
                    i++;
                    return;
                }
                let text = $(this).text();
                if (text === "1") {
                    trailing = true;
                    return;
                }
                title.push(process_text(text));
                i++;
            });
        }
        let j = 0;
        $(this).find('td').each(function () {
            // Take only real columns
            if (j++ < i) {
                data.push(process_text($(this).text()));
            }
        });
        data = data.join(",");
        rows.push(data);
    });
    title = title.join(",");
    rows = rows.join("\n");
    const csv = title + rows;

    // Initiate download action
    const uri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
    const download_link = document.createElement('a');
    download_link.href = uri;
    download_link.download = name + ".csv";
    document.body.appendChild(download_link);
    download_link.click();
    document.body.removeChild(download_link);
}
