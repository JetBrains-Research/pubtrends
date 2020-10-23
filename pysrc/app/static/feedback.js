"use strict";

function clearSelection($buttons) {
    $buttons
        .removeClass('btn-feedback-yes-selected')
        .removeClass('btn-feedback-meh-selected')
        .removeClass('btn-feedback-no-selected');
}

// element - feedback button
//      element should be placed under class btn-group-horizontal, which id is used as key
// value -1, 0, +1
function feedback(element, value) {
    let $element = $(element);
    let groupId = $element.closest('.btn-group-horizontal').attr('id');
    console.info("Feedback " + groupId + ": " + value);
    let $buttons = $('#' + groupId + ' button');
    ['yes', 'meh', 'no'].forEach((opt) => {
        const optClass = 'btn-feedback-' + opt;
        const selectedClass = 'btn-feedback-' + opt + '-selected';
        if ($element.hasClass(optClass)) {
            if ($element.hasClass(selectedClass)) {
                $element.removeClass(selectedClass);
                key = 'cancel:' + key;
            } else {
                clearSelection($buttons);
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
    $.notify("Thanks! You can leave us a message below.", {
        className: "success",
        position: "bottom right"
    });
}

