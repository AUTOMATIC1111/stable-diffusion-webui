{%- extends 'hide.tpl' -%}{% block body %}---
{% if resources.toc != "" and resources.toc != nil %}toc: {{resources.toc}}{% endif %}
{% if resources.title != "" and resources.title != nil %}title: {{resources.title}}{% endif %}
keywords: {{resources.keywords}}
sidebar: home_sidebar
{% if resources.tags != "" and resources.tags != nil %}tags: {{resources.tags}}{% endif %}
{% if resources.summary != "" and resources.summary != nil %}summary: "{{resources.summary}}"{% endif %}
---
{% include 'autogen.tpl' %}

<div class="container" id="notebook-container">
    {{ super()  }}
</div>
{%- endblock body %}
