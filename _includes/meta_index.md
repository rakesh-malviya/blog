<div class="meta_wrapper">
    <div class="list-post-date">
      <time>{{ post.date | date_to_string }}</time>
    </div>

    <div>
    {% if post.tags %}
    {% for tag in post.tags %}
    <a href="{{ site.baseurl }}{{ site.tag_page }}#{{ tag | slugify }}" class="post-tag">{{ tag }}</a>
    {% endfor %}
    {% endif %}

    </div>   
    <p class="desc"> {{ post.excerpt | strip_html | truncate: 100 }} </p>
    <hr>
</div>