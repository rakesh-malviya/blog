<div class="meta_wrapper">
  <div class="post-date">
    <time>{{ page.date | date_to_string }}</time>
  </div>
  <div>
    {% for tag in page.tags %}
    <u><a class="tag_list_link" href="{{ site.baseurl }}/tag/{{ tag }}">{{ tag }}</a></u>
    &nbsp;&nbsp;
    {% endfor %}
  </div>
  <hr>
<div class="meta_wrapper">