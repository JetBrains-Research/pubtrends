"use strict";

/**
 * See for details
 * http://stackoverflow.com/questions/19689715/what-is-the-best-way-to-detect-retina-support-on-a-device-using-javascript
 */
function isHighDensity() {
    return ((window.matchMedia &&
        (window.matchMedia('only screen and (min-resolution: 124dpi), only screen and (min-resolution: 1.3dppx), ' +
            'only screen and (min-resolution: 48.8dpcm)').matches ||
            window.matchMedia('only screen and (-webkit-min-device-pixel-ratio: 1.3), ' +
                'only screen and (-o-min-device-pixel-ratio: 2.6/2), only screen and ' +
                '(min--moz-device-pixel-ratio: 1.3), only screen and (min-device-pixel-ratio: 1.3)').matches)) ||
        (window.devicePixelRatio && window.devicePixelRatio > 1.3));
}

function isSafari() {
    const userAgent = navigator.userAgent;
    return /^((?!chrome|android).)*safari/i.test(userAgent);
}

function process_word_cloud(id, width, height, word_cloud, callback) {
    const word_records = word_cloud['word_records'];
    const ww = word_cloud['width'];
    const wh = word_cloud['height'];
    const canvas = document.getElementById(id);
    const ctx = canvas.getContext("2d");
    // Make canvas visible
    canvas.style.width = width.toString() + "px";
    canvas.style.height = height.toString() + "px";
    const sx = height / wh;
    const sy = width  / ww;

    // Retina/HiDPI fix
    if (isHighDensity()) {
        canvas.width = width * 2;
        canvas.height = height * 2;
        ctx.scale(2, 2);
    } else {
        canvas.width = width;
        canvas.height = height;
    }

    const links = []; // Links information
    let hoverWord = ""; // Word, which cursor points at
    ctx.textBaseline = "top"; // Makes left top point a start point for rendering text

    function addWord(word, x, y, size, vertical, color) {
        ctx.fillStyle = color;
        if (!isSafari()) {
            ctx.font = size.toString() + "px Arial Bold";
        } else {
            ctx.font = size.toString() + "px Arial";
        }

        let linkWidth = ctx.measureText(word).width;
        let linkHeight = parseInt(ctx.font); // Get line height out of font size

        if (vertical) {
            ctx.save();
            ctx.translate(x, y + linkWidth);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(word, 0, 0);
            ctx.restore();
        } else {
            ctx.fillText(word, x, y);
        }

        // Add link params to array for hover processing
        if (vertical) {
            [linkWidth, linkHeight] = [linkHeight, linkWidth];
            links.push([x, y, linkWidth, linkHeight, word]);
        } else {
            links.push([x, y, linkWidth, linkHeight, word]);
        }
    }

    function onMouseMove(ev) {
        // Compute mouse position relative to the canvas in CSS pixels.
        // Using getBoundingClientRect avoids Chrome issues with deprecated layerX/Y
        // and stays correct when the canvas backing store is scaled for HiDPI.
        const rect = canvas.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;

        let hit = false;
        let word = "";

        // Link hover
        for (let i = 0; i < links.length; i++) {
            const link = links[i];
            const linkX = link[0];
            const linkY = link[1];
            const linkWidth = link[2];
            const linkHeight = link[3];
            const linkWord = link[4];

            // Check if cursor is in the link area (account for dx/dy shift used when drawing)
            if (linkX <= x && x  <= linkX + linkWidth &&
                linkY <= y  && y  <= linkY + linkHeight) {
                hit = true;
                word = linkWord;
                break;
            }
        }

        if (hit) {
            document.body.style.cursor = "pointer";
            hoverWord = word;
        } else {
            document.body.style.cursor = "";
            hoverWord = "";
        }
    }

    // Link click
    function onClick(e) {
        if (hoverWord) {
            callback(hoverWord);
        }
    }

    for (let i = 0; i < word_records.length; i++) {
        const wr = word_records[i];
        const word = wr[0], x = wr[1], y = wr[2], size = wr[3], vertical = wr[4], color = wr[5];
        // Add word with small shift, coordinates changed!
        addWord(word, y * sy, x * sx, size, vertical, color);
    }

    // Add mouse listeners
    if (callback != null) {
        canvas.addEventListener("mousemove", onMouseMove, false);
        canvas.addEventListener("click", onClick, false);
    }
}
