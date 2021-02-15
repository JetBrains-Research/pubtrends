"use strict";
// Include this snippet to include notify.js 0.4.2
//     <!-- Notify.js -->
//       <script src="https://cdnjs.cloudflare.com/ajax/libs/notify/0.4.2/notify.min.js"
//               integrity="sha512-efUTj3HdSPwWJ9gjfGR71X9cvsrthIA78/Fvd/IN+fttQVy7XWkOAXb295j8B3cmm/kFKVxjiNYzKw9IQJHIuQ=="
//               crossorigin="anonymous"></script>

function createClickableFeedBackForm(feedBackElementId, feedBackId) {
    createClickableFeedBackFormImpl(feedBackElementId, feedBackId, 'feedback');
}

function createClickableFeedBackFormNoMessage(feedBackElementId, feedBackId) {
    createClickableFeedBackFormImpl(feedBackElementId, feedBackId, 'feedbackNoMessageForm');
}

function createClickableFeedBackFormImpl(feedBackElementId, feedBackId, feedbackFunction) {
    $('#' + feedBackElementId).html(`
        <small className="text-muted">Was this useful?</small>
        <div id="` + feedBackId + `" className="btn-group-horizontal">
            <button type="button" className="btn btn-sm btn-feedback-yes"
                onClick="` + feedbackFunction + `(this, 1)">
                <img src="smile.svg" alt="Yes"/>
            </button>
            <button type="button" className="btn btn-sm btn-feedback-meh"
                onClick="` + feedbackFunction + `(this, 0)">
                <img src="meh.svg" alt="Not sure"/>
            </button>
            <button type="button" className="btn btn-sm btn-feedback-no"
                onClick="` + feedbackFunction + `(this, -1)">
                <img src="frown.svg" alt="No"/>
            </button>
        </div>
    `)
}

// element - feedback button
//      element should be placed under class btn-group-horizontal, which id is used as key
// value -1, 0, +1
function feedback(element, value) {
    feedbackImpl(element, value)
    $.notify("Thanks! You can leave us a message below.", {
        className: "success",
        position: "bottom right"
    });
}

function feedbackNoMessageForm(element, value) {
    feedbackImpl(element, value)
    $.notify("Thanks for the feedback!", {
        className: "success",
        position: "bottom right"
    });
}

function feedbackImpl(element, value) {
    let $element = $(element);
    let feedBackId = $element.closest('.btn-group-horizontal').attr('id');
    console.info("Feedback " + feedBackId + ": " + value);
    let $buttons = $('#' + feedBackId + ' button');
    ['yes', 'meh', 'no'].forEach((opt) => {
        const optClass = 'btn-feedback-' + opt;
        const selectedClass = 'btn-feedback-' + opt + '-selected';
        if ($element.hasClass(optClass)) {
            if ($element.hasClass(selectedClass)) {
                $element.removeClass(selectedClass);
                feedBackId = 'cancel:' + feedBackId;
            } else {
                $buttons
                    .removeClass('btn-feedback-yes-selected')
                    .removeClass('btn-feedback-meh-selected')
                    .removeClass('btn-feedback-no-selected');
                $element.addClass(selectedClass);
            }
        }
    });
    // Decode jobid from URL
    const jobid = new URL(window.location).searchParams.get('jobid');
    $.ajax({
        url: "/feedback",
        type: "POST",
        data: {
            jobid: jobid,
            key: feedBackId,
            value: value
        },

        success: function (data, textStatus, XMLHttpRequest) {
            console.info("Feedback successfully recorded")
        },

        error: function (XMLHttpRequest) {
            const responseText = XMLHttpRequest.responseText;
            console.error("Error recording feedback: " + responseText);
        }
    });
}

