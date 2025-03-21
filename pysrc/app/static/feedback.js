"use strict";
// Please ensure to use jquery.min.js and notify.js for correct functioning
// Include necessary styling options from feedback.css

function createClickableFeedBackForm(feedBackElementId, feedBackId) {
    $('#' + feedBackElementId).html(`
        <small class="text-muted mr-2">Was this useful?</small>
        <div id="` + feedBackId + `" class="btn-group-horizontal">
            <button type="button" class="btn btn-sm btn-feedback-yes"
                onClick="feedback(this, 1)">
                <img src="smile.svg" alt="Yes"/>
            </button>
            <button type="button" class="btn btn-sm btn-feedback-meh"
                onClick="feedback(this, 0)">
                <img src="meh.svg" alt="Not sure"/>
            </button>
            <button type="button" class="btn btn-sm btn-feedback-no"
                onClick="feedback(this, -1)">
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

