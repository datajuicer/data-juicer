from docutils import nodes
from sphinx.transforms import SphinxTransform

VIDEO_LINK_PREFIX = "https://github.com/user-attachments/assets/"

class ReplaceVideoLinksTransform(SphinxTransform):
    default_priority = 900

    def apply(self):

        for node in self.document.traverse(nodes.reference):
            uri = node.get("refuri", "")
            if uri.startswith(VIDEO_LINK_PREFIX):
                video_html = "\n".join([
                    '<video controls width="100%" height="auto" playsinline>',
                    f'  <source src="{uri}" type="video/mp4">',
                    '  Your browser does not support the video tag.',
                    '</video>',
                ])

                raw_node = nodes.raw("", video_html, format="html")
                parent = node.parent
                index = parent.index(node)
                parent.replace(node, raw_node)
