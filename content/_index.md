---
# Leave the homepage title empty to use the site title
title: ''
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: About me
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
    design:
      spacing:
        # Customize the section spacing. Order is top, right, bottom, left.
        padding: ['20px', '0', '20px', '0']
  - block: collection
    id: featured
    content:
      title: Featured Publications
      filters:
        folders:
          - publication
        featured_only: true
    design:
      columns: '1'
      view: compact
      spacing:
        # Customize the section spacing. Order is top, right, bottom, left.
        padding: ['20px', '0', '20px', '0']
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: ''
      text: ''
      # Choose how many pages you would like to display (0 = all pages)
      count: 5
      # Filter on criteria
      filters:
        folders:
          - post
        author: ""
        category: ""
        tag: ""
        exclude_featured: false
        exclude_future: false
        exclude_past: false
        publication_type: ""
      # Choose how many pages you would like to offset by
      offset: 0
      # Page order: descending (desc) or ascending (asc) date.
      order: desc
    design:
      # Choose a layout view
      view: compact
      columns: '2'
      spacing:
        # Customize the section spacing. Order is top, right, bottom, left.
        padding: ['20px', '0', '20px', '0']
  - block: collection
    content:
      title: Publication List
      text: ""
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      view: citation
      spacing:
        # Customize the section spacing. Order is top, right, bottom, left.
        padding: ['20px', '0', '20px', '0']
  - block: experience
    id: timeline
    content:
      title: Timeline
      # Date format for experience
      #   Refer to https://docs.hugoblox.com/customization/#date-format
      date_format: Jan 2006
      # Experiences.
      #   Add/remove as many `experience` items below as you like.
      #   Required fields are `title`, `company`, and `date_start`.
      #   Leave `date_end` empty if it's your current employer.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - title: PhD Candidate
          company: AMLab @ UvA
          company_url: ''
          company_logo: ''
          location: Amsterdam, Netherlands
          date_start: '2024-02-01'
          date_end: ''
          description: |2-
            Under the supervision of Erik Bekkers (University of Amsterdam) and co-supervision of Daniël Pelt (University of Leiden), we will develop techniques for collaborative human-computer image annotation of training sets for deep learning tasks. These techniques will suggest relevant annotations to the human annotator, will indicate inconsistencies in the human annotations, and will use concepts from geometric deep learning to handle shapes of image annotations.
        - title: Research Engineer
          company: Division of Robotics, Perception and Learning @ KTH
          company_url: ''
          company_logo: ''
          location: Stockholm, Sweden
          date_start: '2023-03-01'
          date_end: '2024-02-01'
          description: |2-
              Projects under the supervision of Danica Kragic:

              *  I delved into Geometric Deep Learning and Lie groups while working on a project that involved devising path-finding algorithms on learned equivariant representations through class-pose decomposition.
              *  I also explored Manifold Learning techniques, employing probabilistic models to understand data shapes, following the methodology of Georgios Arvanitidis (2021).
              * Lastly, I worked on a project that proposes a method to recover the underlying hierarchies within hyperbolic embeddings.
        - title: Trainee Internship
          company: Medical Data Analytics Laboratory @ UPM
          company_url: ''
          company_logo: ''
          location: Madrid, Spain
          date_start: '2020-09-01'
          date_end: '2021-01-01'
          description: |2-
            Projects under the supervision of Ernestina Menasalvas Ruiz: 

            * Application of information recognition algorithms in clinical notes.
            * Development of new techniques for the recognition of metrics, doses and numbers.
            * Fine-tuning a BERT model in breast and lung cancer's clinical notes.
    design:
      columns: '2'
      spacing:
        # Customize the section spacing. Order is top, right, bottom, left.
        padding: ['20px', '0', '20px', '0']
#  - block: collection
#    id: featured
#    content:
#      title: Featured Publications
#      filters:
#        folders:
#          - publication
#        featured_only: true
#    design:
#      columns: '2'
#      view: card
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle:
      text: ''
      # Contact (add or remove contact options as necessary)
      email: a.garciacastellanos@uva.nl
      # Automatically link email and phone or display as text?
      autolink: true
    design:
      columns: '2'
      spacing:
        # Customize the section spacing. Order is top, right, bottom, left.
        padding: ['20px', '0', '20px', '0']
---
