// Questions functionality for semantic search
$(document).ready(function() {
    // Create a clickable feedback form for questions
    createClickableFeedBackForm("feedback-questions-form", "feedback-questions");

    // Handle question submission
    $('#ask-question').click(function() {
        const question = $('#question-input').val().trim();
        if (question) {
            submitQuestion(question);
        }
    });

    // Also submit on the Enter key
    $('#question-input').keypress(function(e) {
        if (e.which === 13) {  // Enter key
            const question = $('#question-input').val().trim();
            if (question) {
                submitQuestion(question);
            }
        }
    });

    function submitQuestion(question) {
        // Show loading indicator
        $('#question-results').show();
        $('#question-results-loading').show();
        $('#question-results-content').empty();

        // Decode jobid from URL
        const jobid = new URL(window.location).searchParams.get('jobid');

        // Make AJAX request to the server
        $.ajax({
            url: '/question',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                question: question,
                jobid: jobid
            }),
            success: function(response) {
                // Hide loading indicator
                $('#question-results-loading').hide();
                
                // Display results
                if (response.papers && response.papers.length > 0) {
                    const foundHtml = '<p>Found ' + response.papers.length + ' relevant papers:</p>';
                    const tableStartHtml =
                        '<table id="questions-table" class="table table-sm table-bordered table-striped">\n' +
                        '<thead>' +
                        '<tr>' +
                        '   <th scope="col">#</th>' +
                        '   <th scope="col">Paper</th>' +
                        '   <th scope="col">Authors</th>' +
                        '   <th scope="col">Journal</th>' +
                        '   <th scope="col">Year</th>' +
                        '</tr>' +
                        '</thead>' +
                        '<tbody>'
                    let i = 1;
                    let tableContentHtml = ''
                    for (const paper of response.papers) {
                        tableContentHtml +=
                            '<tr>' +
                            '<th><p class="text-muted small">' + i.toString() + '</p></th>' +
                            '<td>' +
                            '   <a class="fas fa-square" style="color:' + paper.color + '" href="#topic-' + paper.topic + '"></a>&nbsp;' +
                            '   <a href="' + paper.url + '" target="_blank" style="color: black">' + paper.title + '</a>' +
                            '   <p class="text-muted small">' + paper.chunk + '</p>' +
                            '</td>' +
                            '<td><p class="text-muted small">' + paper.authors + '</p></td>' +
                            '<td><p class="text-muted small">' + paper.journal + '</p></td>' +
                            '<td><p class="text-muted small">' + paper.year + '</p></td>' +
                            '</tr>';
                        i += 1;
                    }
                    const tableEndHtml =
                        '</tbody>' +
                        '</table>'
                    const resultsHtml = foundHtml + tableStartHtml + tableContentHtml + tableEndHtml;
                    $('#question-results-content').html(resultsHtml);
                    $('#questions-table').DataTable();
                } else {
                    $('#question-results-content').html('<p>No relevant information found for your question.</p>');
                }
            },
            error: function(xhr, status, error) {
                // Hide loading indicator
                $('#question-results-loading').hide();
                
                // Display error
                $('#question-results-content').html('<div class="alert alert-danger">' +
                    'Error processing your question: ' + error + '</div>');
            }
        });
    }
});