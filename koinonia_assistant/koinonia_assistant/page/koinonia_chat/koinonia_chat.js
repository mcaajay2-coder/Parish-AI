frappe.pages['koinonia-chat'].on_page_load = function(wrapper) {
    var page = frappe.ui.make_app_page({
        parent: wrapper,
        title: 'Koinonia Assistant',
        single_column: true
    });

    // Strip all padding/margin so the iframe fills the full available space
    $(wrapper).find('.layout-main-section-wrapper').css({
        "padding": "0",
        "margin": "0",
        "height": "calc(100vh - 60px)"
    });
    $(wrapper).find('.layout-main-section').css({
        "padding": "0",
        "margin": "0",
        "height": "100%",
        "border": "none",
        "border-radius": "0",
        "box-shadow": "none"
    });
    $(wrapper).find('.page-body').css({
        "padding": "0",
        "height": "100%"
    });
    $(wrapper).find('.page-head').hide();

    // Embed the standalone KOINONIA chat page in a full-height iframe
    page.main.html(`
        <iframe
            src="/koinonia_chat"
            style="
                width: 100%;
                height: calc(100vh - 60px);
                border: none;
                display: block;
                background: #0a0d16;
            "
            frameborder="0"
            allowfullscreen
            allow="microphone *; camera *; clipboard-write *; autoplay *; display-capture *;"
        ></iframe>
    `);
}
