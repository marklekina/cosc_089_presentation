---
title: DZslides template for Pandoc
author: Peter Conrad
date: 26 November 2019
---

<!-- markdownlint-disable MD025 -->

# Section titles

- H1 or H2
- Centered on slide
- Not much room below

<!-- 
  An H1 or H2 renders as a large title in the middle of the slide.
  There is room for a small number of bullets below, but it looks
  nicer with the title alone.
-->

---

How to do a slide

- Use \-\-\- to separate slides
- Use regular text and bullets
- H1 or H2 for a section title

<!--
  Regular text starts closer to the top of the slide.
  A normal text phrase plus bullets makes for a simple,
  attractive slide.
-->

---

# Images

---

You can use images.

- Provide width and height
- Keep them with the HTML file
![meh](../figures/images.jpeg){width=33% height=33%}

<!--
  If you omit width and height, the images tend to
  appear pixel-for-pixel at the resolution of the screen.
  This often means: very huge. Pandoc can resize the
  images for you.
  
  Remember that you need to keep the image files with your
  presentation's HTML file or they won't show up.
-->

---

![Full-screen image with alt text](../figures/images.jpeg){width=100% height=100%}

<!--
  For some reason, a full-screen image renders properly even if
  you omit the width and height tags. I have left them in to foster
  good habits.
-->

---

# Columns

---

2 Columns

:::::::::::::: {.columns}
::: {.column width="50%"}

![meh](../figures/images.jpeg){width=100% height=100%}

:::
::: {.column width="50%"}

- Bullet
- Bullet
- Bullet

<!-- 100% of this column, that is -->

:::
::::::::::::::

<!--
  Pandoc uses fenced div for multiple columns in slide shows.
  I get the impression that DZslides is designed to create
  a slide show with a very simple, uncluttered look.
  If you are using a lot of columns, you might consider
  a different slide format.
-->

---

> Blockquotes look like this

---

Incremental "build" slides

::: incremental

- Incremental slides work
- This is how they look
- It's fine

:::

---

A slide with a pause can work.

. . .

Or can it?

---

# Thank you
