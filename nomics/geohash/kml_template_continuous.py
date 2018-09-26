header_cont="""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<Document>
    <name>{title}</name>
"""

raw_style_cont = """
    <StyleMap id="{name}">
        <Pair>
            <key>normal</key>
            <styleUrl>#{name}1</styleUrl>
        </Pair>
        <Pair>
            <key>highlight</key>
            <styleUrl>#{name}2</styleUrl>
        </Pair>
    </StyleMap>
    <Style id="{name}1">
        <IconStyle>
            <scale>1.1</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>
            </Icon>
            <hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>
        </IconStyle>
        <PolyStyle>
            <color>{color}</color>
        </PolyStyle>
    </Style>
    <Style id="{name}2">
        <IconStyle>
            <scale>1.3</scale>
            <Icon>
                <href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href>
            </Icon>
            <hotSpot x="20" y="2" xunits="pixels" yunits="pixels"/>
        </IconStyle>
        <PolyStyle>
            <color>{color}</color>
        </PolyStyle>
    </Style>
    """

footer_cont = """</Document>
</kml>"""

box_template_cont = """
<Placemark>
        <name>{name}</name>
        <styleUrl>#{style}</styleUrl>
        <Polygon>
            <extrude>1</extrude>
            <tessellate>1</tessellate>
            <altitudeMode>relativeToGround</altitudeMode>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                       {coordinates} 
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>"""