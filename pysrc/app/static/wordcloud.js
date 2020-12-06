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

function process_word_cloud(id, width, height, ww, wh, words, callback) {
    const canvas = document.getElementById(id);
    const ctx = canvas.getContext("2d");
    // Make canvas visible
    canvas.style.width = width.toString() + "px";
    canvas.style.height = height.toString() + "px";
    const sx = (height - 20) / wh;
    const sy = (width - 20) / ww;

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
    var hoverWord = ""; // Word which cursor points at
    ctx.textBaseline = "top"; // Makes left top point a start point for rendering text

    function addWord(word, x, y, size, vertical, color) {

        ctx.fillStyle = color;
        ctx.font = size.toString() + "px Arial Bold";

        var linkWidth = ctx.measureText(word).width,
            linkHeight = parseInt(ctx.font); // Get line height out of font size

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
        var x, y;

        // Get the mouse position relative to the canvas element
        if (ev.layerX || ev.layerX === 0) { // For Firefox
            x = ev.layerX;
            y = ev.layerY;
        }

        // Link hover
        for (let i = 0; i < links.length; i++) {
            const link = links[i];
            const linkX = link[0],
                linkY = link[1],
                linkWidth = link[2],
                linkHeight = link[3],
                linkWord = link[4];

            // Check if cursor is in the link area
            if (linkX <= x && x <= (linkX + linkWidth) && linkY <= y && y <= (linkY + linkHeight)) {
                document.body.style.cursor = "pointer";
                hoverWord = linkWord;
                break;
            } else {
                document.body.style.cursor = "";
                hoverWord = "";
            }
        }
    }

    // Link click
    function onClick(e) {
        if (hoverWord) {
            callback(hoverWord);
        }
    }

    for (let i = 0; i < words.length; i++) {
        const w = words[i];
        const word = w[0], x = w[1], y = w[2], size = w[3], vertical = w[4], color = w[5];
        // Add word with small shift, coordinates changed!
        addWord(word, y * sy + 10, x * sx + 10, size, vertical, color);
    }

    // Add mouse listeners
    if (callback != null) {
        canvas.addEventListener("mousemove", onMouseMove, false);
        canvas.addEventListener("click", onClick, false);
    }
}
