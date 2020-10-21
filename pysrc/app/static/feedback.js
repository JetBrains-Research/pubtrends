"use strict";

// element - feedback button
//      element should be placed under class btn-group-horizontal, which id is used as key
// value -1, 0, +1
function feedback(element, value) {
    let $element = $(element);
    let groupId = $element.closest('.btn-group-horizontal').attr('id');
    console.info("Feedback " + groupId + ": " + value);
    // Deselect all
    let $buttons = $('#' + groupId + ' button');
    $buttons
        .removeClass('btn-feedback-yes-selected')
        .removeClass('btn-feedback-meh-selected')
        .removeClass('btn-feedback-no-selected');
    // Select only one
    if ($element.hasClass('btn-feedback-yes')) {
        $element.addClass('btn-feedback-yes-selected');
    }
    if ($element.hasClass('btn-feedback-meh')) {
        $element.addClass('btn-feedback-meh-selected');
    }
    if ($element.hasClass('btn-feedback-no')) {
        $element.addClass('btn-feedback-no-selected');
    }
    // Decode jobid from URL
    const jobid = new URL(window.location).searchParams.get('jobid');
    $.ajax({
        url: "/feedback",
        type: "POST",
        data: {
            jobid: jobid,
            key: groupId,
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
    $.notify("Thank you for the feedback!", {
        className: "success",
        position: "bottom right"
    });
}

