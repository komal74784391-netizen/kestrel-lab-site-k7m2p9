---
layout: default
title: Home
---

# Kestrel Computational Biology Lab

We develop computational methods for single-cell genomics, spatially resolved
transcriptomics and structural biology.  The lab is led by Dr. Alexander
Roberts at Ashford University.

## Publications

{% for pub in site.publications reversed %}
- [{{ pub.title }}]({{ pub.url }}) — {{ pub.venue }} ({{ pub.year }})
{% endfor %}
