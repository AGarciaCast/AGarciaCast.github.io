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
      title: Publications
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      columns: '1'
      view: compact
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
        - title: Research Engineer
          company: Division of Robotics, Perception and Learning @ KTH
          company_url: ''
          company_logo: ''
          location: Stockholm, Sweden
          date_start: '2023-03-01'
          date_end: ''
          description: |2-
              Projects under the supervision of Danica Kragic:

              *  I delved into Geometric Deep Learning and Lie groups while working on a project that involved devising path-finding algorithms on learned equivariant representations through class-pose decomposition.
              *  I also explored Manifold Learning techniques, employing probabilistic models to understand data shapes, following the methodology of Georgios Arvanitidis (2021).
              * Currently, I am working on a project that aims to recover the underlying hierarchies within hyperbolic embeddings.
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
      email: algc@kth.se
      # Automatically link email and phone or display as text?
      autolink: true
    design:
      columns: '2'
---
