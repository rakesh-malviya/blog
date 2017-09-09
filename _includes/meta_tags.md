<div class="meta_wrapper">
    <div class="list-post-date">
      <time>{{ post.date | date_to_string }}</time>
    </div>   
    <p class="desc"> {{ post.excerpt | strip_html | truncate: 100 }} </p>
    <hr>
</div>