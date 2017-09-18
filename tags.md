---
layout: default
title: Tags
---
<header class="masthead">
  <h1 class="masthead-title--small">
    <a href="{{ site.baseurl }}/">{{ site.name }}</a>
  </h1>
</header>      
<div class="tags-expo">
  <div class="tags-expo-list">
    {% for tag in site.tags %}
    <a href="#{{ tag[0] | slugify }}" class="post-tag">{{ tag[0] }}</a>
    {% endfor %}
  </div>
  <hr/>
  <div class="tags-expo-section">
    {% for tag in site.tags %}
    <h2 id="{{ tag[0] | slugify }}">{{ tag[0] }}</h2>
    <ul class="tags-expo-posts">
      {% for post in tag[1] %}
      <li>         
          <a href="{{ site.baseurl }}{{ post.url }}">
          {{ post.title }}</a>
          {% if post.date %}{% include meta_tags.md %}{% endif %}
        </li>
      {% endfor %}
    </ul>
    {% endfor %}
  </div>
</div>
{% include analytics.md %}
