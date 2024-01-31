{%- extends 'basic.tpl' -%}

{% block input_group -%}
{%- if cell.metadata.hide_input or nb.metadata.hide_input -%}
{%- else -%}
    {{ super()  }}
{%- endif -%}
{% endblock input_group %}

{% block output_group -%}
{%- if cell.metadata.hide_output -%}
{%- else -%}
    {{ super()  }}
{%- endif -%}
{% endblock output_group %}

{% block output_area_prompt %}
{%- if cell.metadata.hide_input or nb.metadata.hide_input -%}
   <div class="prompt"> </div>
{%- else -%}
    {{ super()  }}
{%- endif -%}
{% endblock output_area_prompt %}
