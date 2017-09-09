<div class="meta_wrapper">
  <span class="post-date">{{ page.date | date_to_string }}</span>
  {% if page.tags %}
  {% for tag in page.tags %}
    <a href="{{ site.baseurl }}{{ site.tag_page }}#{{ tag | slugify }}" class="post-tag">{{ tag }}</a>
  {% endfor %}
  {% endif %}
<div class="meta_wrapper">