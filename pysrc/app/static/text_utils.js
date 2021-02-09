function normalize(text) {
    return text.replace(/&#34;/g, '"').replace(/&#39;/g, '\'').replace(/&amp;/g, '&');
}