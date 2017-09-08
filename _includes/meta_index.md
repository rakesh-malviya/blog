<div class="meta_wrapper">
    <div class="list-post-date">
      <time>{{ post.date | date_to_string }}</time>
    </div>
    <div>
      {% for tag in post.tags %}
          <u><a class="tag_list_link" href="{{ site.baseurl }}/tag/{{ tag }}">{{ tag }}</a></u>
          &nbsp;&nbsp;
      {% endfor %}
    </div>   
    <p class="desc"> {{ post.excerpt | strip_html | truncate: 100 }} </p>
    <hr>
</div>