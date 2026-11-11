"""
generate_layer5_heldout.py — Phase 4 (Refined with High Diversity & Balanced Labels)
Generates:
  1. dataset/layer5_heldout/heldout_responses.csv (200 unique AI responses)
  2. dataset/layer5_heldout/heldout_claims.csv (~850-950 unique atomic claims)
  3. dataset/layer5_heldout/heldout_evidence.csv (5 candidate passages per claim)
"""

import os, csv, random

random.seed(2026)

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
OUT_DIR = os.path.join(BASE, "layer5_heldout")
os.makedirs(OUT_DIR, exist_ok=True)

TOPICS = [
    "Technology", "Science", "History", "Geography", "Politics and government",
    "Economics", "Sports", "Medicine and health", "Environment", "Education",
    "Space", "Companies and organizations", "Famous people", "Culture", "General knowledge"
]

RESPONSE_STYLES = [
    "explanatory_paragraph", "bullet_points", "comparison", "short_answer",
    "historical_narrative", "technical", "conversational"
]

# We build 200 distinct topics/prompts with unique claims each
SEED_ENTITIES = [
    # (topic, entity, (c_text, ctype, label, diff, htype, gold_ev, partial_ev))
    # Tech
    ("Technology", "Python language", [
        ("Python was created by Guido van Rossum.", "person", "SUPPORTS", "easy", "none",
         "Guido van Rossum began developing Python in 1989 and released version 0.9.0 in 1991.", "Python is widely used in data science."),
        ("Python was first released in 1991.", "date", "SUPPORTS", "easy", "none",
         "Python was released in February 1991 as an open-source programming language.", "Early scripting languages emerged in the late 20th century."),
        ("Python was developed at Bell Labs in Murray Hill.", "location", "REFUTES", "medium", "wrong_location",
         "Python was developed at CWI in Amsterdam, Netherlands, not Bell Labs.", "Bell Labs created Unix and the C programming language."),
        ("Python 3.0 was released in December 2008.", "date", "NOT_ENOUGH_INFO", "hard", "none",
         "Python 3.0 introduced significant changes to eliminate redundant syntax constructs.", "Major software updates are released periodically.")
    ]),
    ("Technology", "Linux kernel", [
        ("Linus Torvalds released the first Linux kernel in 1991.", "person", "SUPPORTS", "easy", "none",
         "Linus Torvalds posted the first release of the Linux kernel to the comp.os.minix newsgroup on September 17, 1991.", "Operating system kernels manage hardware resources."),
        ("Linux kernel was written entirely in Python.", "scientific", "REFUTES", "medium", "incorrect_technical_specification",
         "The Linux kernel is written predominantly in C and assembly language, not Python.", "Programming languages are chosen based on system performance requirements."),
        ("Linux is released under the GNU General Public License version 2.", "organization", "SUPPORTS", "medium", "none",
         "Linus Torvalds adopted the GNU GPL version 2 license for the Linux kernel in 1992.", "Open source licenses define terms of software redistribution."),
        ("Linux was originally developed at Stanford University.", "location", "REFUTES", "medium", "wrong_location",
         "Linus Torvalds created Linux while studying at the University of Helsinki in Finland.", "University computer science departments often contribute to operating system research.")
    ]),
    ("Technology", "World Wide Web", [
        ("Tim Berners-Lee invented the World Wide Web in 1989.", "person", "SUPPORTS", "easy", "none",
         "Tim Berners-Lee proposed the World Wide Web in March 1989 while working at CERN.", "CERN is a major European particle physics research centre."),
        ("The first website went live in August 1991.", "date", "SUPPORTS", "medium", "none",
         "The first web page was served on August 6, 1991 at CERN.", "Web servers deliver hypertext documents over IP networks."),
        ("CERN patented the World Wide Web protocols for commercial licensing.", "organization", "REFUTES", "medium", "negation_error",
         "CERN made the World Wide Web software available in the public domain on a royalty-free basis in 1993.", "Intellectual property management can influence technology adoption."),
        ("Tim Berners-Lee developed the Web using a NeXT computer.", "scientific", "NOT_ENOUGH_INFO", "hard", "none",
         "Tim Berners-Lee designed hypertext transfer protocols and document format specifications.", "NeXT computers were workstations developed in the late 1980s.")
    ]),
    ("Technology", "JavaScript language", [
        ("Brendan Eich developed JavaScript in 1995.", "person", "SUPPORTS", "easy", "none",
         "Brendan Eich created JavaScript in May 1995 while working at Netscape Communications Corporation.", "Web scripting languages run inside client web browsers."),
        ("JavaScript was developed over a period of ten years before release.", "date", "REFUTES", "medium", "wrong_number",
         "Brendan Eich developed the initial JavaScript prototype in just 10 days in May 1995.", "Rapid prototyping enabled rapid web standard development."),
        ("ECMA International standardizes JavaScript under ECMAScript.", "organization", "SUPPORTS", "easy", "none",
         "JavaScript standard specifications are maintained by ECMA International under the ECMAScript standard.", "Standards organizations publish technical specifications."),
        ("JavaScript requires Java Virtual Machine runtime execution.", "definition", "REFUTES", "hard", "entity_substitution",
         "JavaScript is an independent scripting language with syntax inspired by C and Java, but does not use the JVM.", "Virtual machines provide sandboxed bytecode execution environments.")
    ]),
    ("Technology", "Transistor invention", [
        ("The transistor was invented at Bell Labs in 1947.", "location", "SUPPORTS", "easy", "none",
         "John Bardeen, Walter Brattain, and William Shockley invented the point-contact transistor at Bell Labs in December 1947.", "Telecommunications labs conducted solid-state physics experiments."),
        ("The transistor was invented by Thomas Edison in Menlo Park.", "person", "REFUTES", "easy", "wrong_person",
         "The transistor was invented by Bardeen, Brattain, and Shockley at Bell Labs, not by Thomas Edison.", "Thomas Edison developed incandescent lighting and electrical distribution systems."),
        ("Transistors replaced bulky thermionic vacuum tubes.", "cause_effect", "SUPPORTS", "medium", "none",
         "Solid-state transistors replaced vacuum tubes, leading to dramatic miniaturization and lower power consumption in electronics.", "Miniaturization paved the way for modern computing devices."),
        ("The 1956 Nobel Prize in Physics was awarded for the transistor.", "event", "NOT_ENOUGH_INFO", "hard", "none",
         "The invention of the transistor transformed global electronic circuit manufacturing.", "Nobel prizes are awarded annually for outstanding scientific discoveries.")
    ]),
]

# Generate rich seeds algorithmically across all topics
TOPIC_BASE_DATA = {
    "Technology": [
        ("C programming language", "Dennis Ritchie", "Bell Labs", "1972", "systems programming", "high-level language"),
        ("Relational database", "Edgar F. Codd", "IBM Research", "1970", "relational algebra", "hierarchical model"),
        ("Ethernet networking", "Robert Metcalfe", "Xerox PARC", "1973", "packet switching", "token ring"),
        ("Flash memory", "Fujio Masuoka", "Toshiba", "1984", "floating gate transistors", "magnetic tape"),
        ("Git version control", "Linus Torvalds", "Open Source Community", "2005", "distributed versioning", "centralized server"),
        ("Public key crypto", "Whitfield Diffie", "Stanford University", "1976", "asymmetric keys", "symmetric cipher"),
        ("ARM architecture", "Sophie Wilson", "Acorn Computers", "1985", "RISC instructions", "CISC instructions"),
        ("Solid state drives", "NAND flash", "storage industry", "1991", "semiconductor storage", "spinning disks"),
        ("Android operating system", "Andy Rubin", "Android Inc", "2003", "Linux kernel", "Windows NT"),
        ("Quantum computing", "David Deutsch", "Oxford University", "1985", "qubit superposition", "binary logic"),
    ],
    "Science": [
        ("Speed of light", "299792458 m/s", "vacuum", "fundamental constant", "relativity", "sound wave"),
        ("DNA double helix", "Watson and Crick", "Cambridge", "1953", "nucleotide base pairing", "protein triple helix"),
        ("Periodic table", "Dmitri Mendeleev", "Saint Petersburg", "1869", "atomic weights", "alphabetical order"),
        ("Photosynthesis", "chlorophyll pigments", "chloroplasts", "light energy conversion", "glucose and oxygen", "dark reaction"),
        ("CRISPR gene editing", "Jennifer Doudna", "UC Berkeley", "2012", "Cas9 molecular scissors", "radiation mutagenesis"),
        ("Mitochondria", "ATP synthesis", "eukaryotic cells", "endosymbiotic origin", "cellular respiration", "nuclear replication"),
        ("Gravitational waves", "LIGO detectors", "Livingston and Hanford", "2015", "black hole merger", "electromagnetic signal"),
        ("Higgs boson", "CERN LHC", "Geneva", "2012", "mass generation field", "dark matter particle"),
        ("Electron discovery", "J. J. Thomson", "Cavendish Laboratory", "1897", "cathode ray deflection", "atomic nucleus"),
        ("Plate tectonics", "Alfred Wegener", "continental drift", "1912", "lithospheric plates", "expanding earth"),
    ],
    "History": [
        ("Apollo 11 Moon landing", "Neil Armstrong", "Sea of Tranquility", "1969", "crewed lunar landing", "Mars rover"),
        ("French Revolution", "Storming of Bastille", "Paris", "1789", "constitutional monarchy overthrow", "industrial uprising"),
        ("Magna Carta", "King John", "Runnymede", "1215", "feudal rights and legal limits", "declaration of war"),
        ("Fall of Berlin Wall", "Gunter Schabowski", "Berlin", "1989", "German reunification", "Cold War escalation"),
        ("Treaty of Versailles", "Woodrow Wilson and Allies", "Hall of Mirrors", "1919", "World War I peace treaty", "League of Nations abolition"),
        ("United Nations founding", "San Francisco conference", "United States", "1945", "multilateral peace charter", "bilateral trade pact"),
        ("Industrial Revolution", "James Watt", "Great Britain", "1760", "steam power and mechanization", "hunter-gatherer transition"),
        ("Suez Canal opening", "Ferdinand de Lesseps", "Egypt", "1869", "maritime shortcut", "railway network"),
        ("Printing press", "Johannes Gutenberg", "Mainz", "1440", "movable metal type", "handwritten manuscripts"),
        ("Ancient Olympics", "Greek athletes", "Olympia", "776 BC", "religious athletic festival", "military invasion"),
    ],
    "Geography": [
        ("Mount Everest", "8848.86 metres", "Nepal and China border", "Himalayas", "highest mountain above sea level", "deepest ocean trench"),
        ("Amazon River", "209000 m3/s discharge", "South America", "Atlantic Ocean drainage", "largest river discharge volume", "longest European canal"),
        ("Mariana Trench", "10994 metres", "Western Pacific", "Challenger Deep", "deepest ocean point", "shallow lake"),
        ("Lake Baikal", "1642 metres depth", "Siberia Russia", "Rift valley lake", "deepest freshwater lake", "man-made reservoir"),
        ("Sahara Desert", "9200000 sq km", "North Africa", "hyper-arid desert", "largest hot desert", "tropical rainforest"),
        ("Great Barrier Reef", "2300 km length", "Queensland Australia", "Coral Sea", "largest coral reef system", "polar ice shelf"),
        ("Dead Sea", "430 metres below sea level", "Jordan Valley", "hypersaline lake", "lowest land elevation", "mountain plateau"),
        ("Nile River", "6650 kilometres", "Northeastern Africa", "Mediterranean Sea", "longest river system", "mountain brook"),
        ("Mount Kilimanjaro", "5895 metres", "Tanzania", "stratovolcano", "highest peak in Africa", "underwater ridge"),
        ("Victoria Falls", "1708 metres wide", "Zambia and Zimbabwe", "Zambezi River", "largest sheet of falling water", "coastal bay"),
    ],
    "Politics and government": [
        ("Bicameral legislature", "two chambers", "parliamentary and presidential systems", "checks and balances", "bicameral system", "unicameral assembly"),
        ("Universal Declaration of Human Rights", "UN General Assembly", "Paris", "1948", "fundamental rights charter", "military alliance"),
        ("US Constitution", "Constitutional Convention", "Philadelphia", "1787", "federal government structure", "royal decree"),
        ("International Court of Justice", "principal UN judicial organ", "The Hague Netherlands", "1945", "interstate legal disputes", "criminal court for individuals"),
        ("European Parliament", "direct EU elections", "Strasbourg and Brussels", "1979", "supranational legislative assembly", "national congress"),
        ("Geneva Conventions", "humanitarian law", "Geneva Switzerland", "1949", "protection of non-combatants", "customs union"),
        ("Supreme Court of the US", "nine justices", "Washington DC", "Article III", "highest appellate jurisdiction", "executive department"),
        ("Separation of powers", "Montesquieu", "tripartite government", "1748", "legislative executive judicial", "absolute autocracy"),
        ("Federalism", "shared sovereignty", "federal and state authorities", "constitutional division", "multi-level governance", "unitary state"),
        ("Judicial review", "Marbury v Madison", "Supreme Court", "1803", "constitutional invalidation power", "tax collection mechanism"),
    ],
    "Economics": [
        ("Federal Reserve System", "central bank", "United States", "1913", "monetary policy and inflation", "private retail store"),
        ("Gross Domestic Product", "final goods and services", "national boundaries", "economic measurement", "total economic output", "stock market index"),
        ("Comparative advantage", "David Ricardo", "Principles of Political Economy", "1817", "lower opportunity cost trade", "absolute trade embargo"),
        ("Bretton Woods Conference", "IMF and World Bank", "New Hampshire", "1944", "international monetary framework", "colonial treaty"),
        ("Inflation", "general price level increase", "purchasing power decline", "consumer price index", "macroeconomic expansion", "deflationary spiral"),
        ("Keynesian economics", "John Maynard Keynes", "General Theory", "1936", "aggregate demand stabilization", "laissez-faire abstention"),
        ("Price elasticity of demand", "quantity responsiveness", "price variations", "microeconomic analysis", "demand sensitivity", "fixed quantity quota"),
        ("Gini coefficient", "Corrado Gini", "wealth distribution", "1912", "inequality statistical metric", "employment rate"),
        ("Tariff policy", "customs duty on imports", "international trade barriers", "domestic protectionism", "import taxation", "free trade exemption"),
        ("European Central Bank", "Frankfurt Germany", "Eurozone monetary policy", "1998", "euro currency management", "national commercial bank"),
    ],
    "Sports": [
        ("Modern Olympic Games", "Pierre de Coubertin", "Athens Greece", "1896", "multisport international games", "regional tournament"),
        ("FIFA World Cup", "Jules Rimet", "Uruguay", "1930", "men's association football championship", "club exhibition match"),
        ("Standard marathon", "42.195 kilometres", "London Olympics", "1908", "official long-distance running", "sprint race"),
        ("Basketball invention", "James Naismith", "Springfield Massachusetts", "1891", "indoor team sport", "individual martial art"),
        ("Wimbledon Championship", "All England Club", "London", "1877", "grass court tennis tournament", "indoor squash event"),
        ("Tour de France", "Geo Lefevre", "France", "1903", "multi-stage bicycle race", "motorcycle rally"),
        ("World 100m Record", "Usain Bolt", "Berlin Germany", "2009", "9.58 seconds sprint record", "marathon split time"),
        ("Formula One Championship", "Silverstone Circuit", "United Kingdom", "1950", "premier open-wheel auto racing", "karting competition"),
        ("Winter Olympic Games", "Chamonix", "France", "1924", "international snow and ice events", "summer track meet"),
        ("Rugby World Cup", "International Rugby Board", "New Zealand and Australia", "1987", "rugby union tournament", "cricket series"),
    ],
    "Medicine and health": [
        ("Penicillin discovery", "Alexander Fleming", "St Mary's Hospital London", "1928", "first antibacterial antibiotic", "viral vaccine"),
        ("Smallpox eradication", "World Health Organization", "global vaccination campaign", "1980", "complete disease eradication", "seasonal outbreak"),
        ("Insulin isolation", "Banting and Best", "University of Toronto", "1921", "pancreatic hormone purification", "synthetic vitamin"),
        ("Polio vaccine", "Jonas Salk", "University of Pittsburgh", "1955", "inactivated poliovirus inoculation", "antibiotic capsule"),
        ("Human Genome Project", "international consortium", "Department of Energy and NIH", "2003", "human DNA sequencing", "protein crystallization"),
        ("Tuberculosis", "Robert Koch", "Mycobacterium tuberculosis", "1882", "bacterial lung infection", "viral influenza"),
        ("X-rays discovery", "Wilhelm Rontgen", "Wurzburg University", "1895", "electromagnetic radiation imaging", "ultrasound wave"),
        ("Aspirin synthesis", "Felix Hoffmann", "Bayer Germany", "1897", "acetylsalicylic acid", "opium derivative"),
        ("Type 1 diabetes", "autoimmune destruction", "beta cells of pancreas", "insulin deficiency", "chronic endocrine condition", "bacterial contagion"),
        ("Hemoglobin protein", "four polypeptide chains", "erythrocytes", "oxygen transport molecule", "iron-binding protein", "digestive enzyme"),
    ],
    "Environment": [
        ("Paris Climate Agreement", "196 national parties", "COP 21 Paris", "2015", "limiting global temperature rise", "free trade zone"),
        ("Montreal Protocol", "United Nations Environment Programme", "Montreal Canada", "1987", "CFC ozone protection treaty", "ocean fishing quota"),
        ("Kyoto Protocol", "UNFCCC parties", "Kyoto Japan", "1997", "legally binding emission reductions", "voluntary conservation pledge"),
        ("Amazon Rainforest", "5500000 sq km", "South America", "carbon dioxide absorption", "largest tropical rainforest carbon sink", "temperate pine forest"),
        ("Ocean acidification", "atmospheric CO2 absorption", "carbonic acid formation", "marine pH decline", "coral reef calcification stress", "ocean warming"),
        ("IPCC", "WMO and UNEP", "Geneva Switzerland", "1988", "climate change scientific assessments", "wildlife sanctuary operator"),
        ("Coral bleaching", "thermal stress", "zooxanthellae expulsion", "marine ecosystem impact", "algae symbiotic disruption", "nutrient enrichment"),
        ("Wetlands conservation", "Ramsar Convention", "Ramsar Iran", "1971", "wetland ecosystem protection", "dam building initiative"),
        ("Greenhouse effect", "greenhouse gas absorption", "infrared planetary radiation", "tropospheric warming", "thermal equilibrium maintenance", "solar flare deflection"),
        ("Deforestation", "land clearance for agriculture", "tropical regions", "carbon emission release", "biodiversity habitat loss", "soil enrichment"),
    ],
    "Education": [
        ("Formative assessment", "ongoing diagnostic evaluation", "instructional period", "learning feedback", "pedagogical adjustment", "final licensing exam"),
        ("Summative assessment", "end-of-unit evaluation", "benchmark comparison", "grading and certification", "cumulative achievement measurement", "diagnostic warmup"),
        ("Bloom's Taxonomy", "Benjamin Bloom", "University of Chicago", "1956", "cognitive domain classification", "psychomotor drill"),
        ("Montessori education", "Maria Montessori", "Rome Italy", "1907", "child-centered self-directed learning", "rote memorization lecture"),
        ("University of Bologna", "medieval scholar guild", "Bologna Italy", "1088", "oldest continuous university", "secondary school academy"),
        ("PISA assessment", "OECD", "15-year-old students", "triennial international testing", "reading math science literacy", "preschool evaluation"),
        ("Zone of Proximal Development", "Lev Vygotsky", "sociocultural theory", "guided instructional scaffolding", "learner developmental potential", "independent mastery level"),
        ("Piaget's stages of development", "Jean Piaget", "four cognitive stages", "constructivist learning", "sensorimotor to formal operational", "behaviorist conditioning"),
        ("Andragogy theory", "Malcolm Knowles", "adult education", "self-directed adult learning", "experiential problem-centered orientation", "early childhood daycare"),
        ("Open Educational Resources", "UNESCO", "public domain or open license", "free digital teaching materials", "barrier-free access to knowledge", "paywalled subscription textbooks"),
    ],
    "Space": [
        ("Mars natural moons", "Phobos and Deimos", "Asaph Hall", "1877", "two small captured asteroid satellites", "four large Galilean moons"),
        ("James Webb Space Telescope", "NASA ESA CSA", "L2 Lagrange point", "2021", "infrared space observatory", "ground-based optical telescope"),
        ("Voyager 1 probe", "NASA JPL", "interstellar space", "2012", "most distant human-made spacecraft", "lunar lander module"),
        ("Pluto dwarf planet", "International Astronomical Union", "Prague resolution", "2006", "dwarf planet classification", "ninth major classical planet"),
        ("Jupiter Great Red Spot", "persistent anticyclonic storm", "southern hemisphere of Jupiter", "telescopic observation", "larger than Earth diameter", "frozen crater basin"),
        ("Apollo 17 mission", "Eugene Cernan and Harrison Schmitt", "Taurus-Littrow valley", "1972", "final crewed Apollo lunar landing", "first Mars orbital mission"),
        ("Curiosity rover", "NASA Mars Science Laboratory", "Gale Crater Mars", "2012", "Martian habitability exploration", "Venus atmospheric probe"),
        ("Saturn ring system", "water ice particles and rock", "planetary rings", "equatorial plane", "extensive celestial ring system", "solid metallic shield"),
        ("Sagittarius A*", "supermassive black hole", "Milky Way galactic center", "4 million solar masses", "compact radio source", "interstellar nebular cloud"),
        ("Halley's Comet", "Edmond Halley", "75 to 76 year orbit", "periodic retrograde comet", "inner solar system return", "intergalactic asteroid"),
    ],
    "Companies and organizations": [
        ("Apple Inc founding", "Jobs Wozniak and Wayne", "Los Altos California", "1976", "personal microcomputer pioneer", "mainframe leasing vendor"),
        ("Microsoft founding", "Bill Gates and Paul Allen", "Albuquerque New Mexico", "1975", "microcomputer BASIC software", "telecom network provider"),
        ("Google founding", "Larry Page and Sergey Brin", "Stanford University", "1998", "PageRank search engine", "hardware manufacturing plant"),
        ("Amazon founding", "Jeff Bezos", "Bellevue Washington", "1994", "online bookstore to cloud provider", "brick-and-mortar grocery"),
        ("International Monetary Fund", "Bretton Woods participants", "Washington DC", "1944", "international financial stability", "private investment bank"),
        ("Red Cross Movement", "Henry Dunant", "Geneva Switzerland", "1863", "humanitarian medical relief", "military alliance command"),
        ("CERN organization", "twelve European states", "Geneva Switzerland", "1954", "particle physics laboratory", "space agency consortium"),
        ("World Trade Organization", "Marrakesh Agreement", "Geneva Switzerland", "1995", "global international trade rules", "customs enforcement agency"),
        ("SpaceX founding", "Elon Musk", "El Segundo California", "2002", "commercial reusable rocketry", "government postal service"),
        ("Amnesty International", "Peter Benenson", "London United Kingdom", "1961", "human rights defense NGO", "political lobbying party"),
    ],
    "Famous people": [
        ("Albert Einstein Nobel Prize", "photoelectric effect law", "Physics prize", "1921", "theoretical physics contribution", "general relativity award"),
        ("Marie Curie Nobel Prizes", "Physics 1903 and Chemistry 1911", "radioactivity research", "Sorbonne professor", "two different scientific fields", "peace prize recipient"),
        ("Sir Isaac Newton", "laws of motion and gravitation", "Principia Mathematica", "1687", "classical physics foundation", "quantum mechanics model"),
        ("Nelson Mandela", "anti-apartheid struggle", "President of South Africa", "1994", "democratic state leadership", "monarchical sovereign"),
        ("Charles Darwin", "natural selection evolution", "On the Origin of Species", "1859", "biological evolutionary theory", "creationist taxonomy"),
        ("Ada Lovelace", "first computer algorithm", "Babbage Analytical Engine", "1843", "early computing pioneer", "electronic transistor inventor"),
        ("Leonardo da Vinci", "Mona Lisa and The Last Supper", "Renaissance Florence", "polymath artist engineer", "masterpiece painter", "modern digital illustrator"),
        ("Mahatma Gandhi", "nonviolent civil resistance", "Indian independence movement", "Salt March 1930", "Satyagraha philosophy", "military general"),
        ("Alan Turing", "Enigma machine decryption", "Bletchley Park", "computational theory", "Turing machine concept", "radio telescope engineer"),
        ("Martin Luther King Jr", "I Have a Dream speech", "March on Washington", "1963", "civil rights leadership", "presidential candidate"),
    ],
    "Culture": [
        ("Japanese haiku", "5-7-5 syllable structure", "nature seasonal theme", "Matsuo Basho", "traditional poetic form", "five-act theatrical play"),
        ("Tango dance", "Rio de la Plata ports", "Buenos Aires and Montevideo", "late 19th century", "partner dance and music", "classical Viennese waltz"),
        ("Impressionism art", "Claude Monet and contemporaries", "Paris France", "1870s", "light and brushstroke painting", "strict neoclassicism"),
        ("Gothic architecture", "pointed arches and flying buttresses", "Abbey of Saint-Denis", "12th century France", "cathedral architecture style", "modernist glass skyscraper"),
        ("Kabuki theatre", "elaborate makeup and costumes", "Kyoto and Edo", "17th century Japan", "stylized classical performance", "silent motion picture"),
        ("Shakespearean sonnet", "14 lines iambic pentameter", "ABAB CDCD EFEF GG scheme", "Elizabethan England", "classical poetic structure", "prose essay"),
        ("Jazz music origins", "African-American community", "New Orleans Louisiana", "early 20th century", "syncopated improvisation", "medieval Gregorian chant"),
        ("Origami art", "paper folding without cuts", "traditional Japanese craft", "geometric animal designs", "sculptural paper folding", "oil painting"),
        ("Cinema invention", "Auguste and Louis Lumiere", "Cinematographe projector", "Paris 1895", "first public movie screening", "sound recording phonograph"),
        ("Flamenco art", "Andalusia region", "southern Spain", "cante baile and toque", "traditional folkloric music and dance", "operatic chorus"),
    ],
    "General knowledge": [
        ("Water freezing and boiling", "0 degrees C freeze 100 degrees C boil", "1 atmosphere pressure", "standard temperature pressure", "phase changes of water", "sublimation only"),
        ("Earth rotation period", "23 hours 56 minutes 4 seconds", "sidereal day", "planetary axis rotation", "diurnal day cycle", "365 hour rotation"),
        ("Leap year rule", "every 4 years except century divisible by 400", "Gregorian calendar", "366 calendar days", "equinoctial alignment", "every 2 years"),
        ("English alphabet", "26 letters", "five vowels and 21 consonants", "Latin script", "standard written characters", "50 hieroglyphic symbols"),
        ("Diamond mineral hardness", "rating 10 on Mohs scale", "carbon crystal lattice", "scratch resistance", "hardest natural mineral", "soft sedimentary rock"),
        ("Atmospheric composition", "78% nitrogen 21% oxygen", "trace noble gases", "troposphere air mixture", "Earth gas envelope", "pure hydrogen gas"),
        ("Prime number definition", "greater than 1 divisible only by 1 and itself", "number theory", "fundamental arithmetic building block", "prime factorization", "even numbers only"),
        ("Sound speed in air", "343 metres per second", "dry air at 20 degrees Celsius", "acoustic pressure wave", "sonic propagation", "light speed equivalence"),
        ("Solar eclipse alignment", "Moon between Earth and Sun", "syzygy orbital alignment", "solar corona visibility", "astronomical occultation", "planetary transit of Venus"),
        ("Primary colors of light", "red green and blue RGB", "additive color mixing", "visible spectrum components", "screen display technology", "cyan magenta yellow subtractive"),
    ]
}

DISTRACTOR_PASSAGES = [
    "The International System of Units defines seven fundamental physical base units.",
    "The Great Barrier Reef is the largest coral reef system on Earth, located in Australia.",
    "The French Revolution began in 1789 with the storming of the Bastille in Paris.",
    "Photosynthesis converts carbon dioxide and water into oxygen and carbohydrate molecules.",
    "Johannes Gutenberg developed movable type printing in Europe around 1440.",
    "The Mona Lisa is a Renaissance portrait painting by Leonardo da Vinci displayed at the Louvre.",
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
    "The human heart is a muscular organ that pumps blood throughout the circulatory system.",
    "The Magna Carta was sealed by King John of England in June 1215.",
    "The periodic table arranges chemical elements by increasing atomic number.",
    "The Suez Canal connects the Mediterranean Sea directly to the Red Sea.",
    "DNA is organized into long structures called chromosomes within eukaryotic cell nuclei.",
    "The Federal Reserve acts as the central banking authority in the United States.",
    "Saturn possesses a prominent and extensive planetary ring system composed of water ice.",
    "Usain Bolt holds the world record in the 100-metre sprint at 9.58 seconds.",
    "The Amazon River in South America has the highest water discharge volume of any river.",
    "The human brain contains approximately 86 billion neurons interconnected by synapses.",
    "The speed of light in vacuum is approximately 300,000 kilometres per second.",
    "The European Union is an economic and political union of 27 member states.",
    "The International Court of Justice is seated in The Hague, Netherlands."
]

def get_response_verdict(s_count, r_count, n_count, total):
    if s_count == total:
        return "Fully factual"
    elif s_count >= total * 0.8 and r_count == 0:
        return "Mostly factual"
    elif r_count > total * 0.5:
        return "Mostly inaccurate"
    elif r_count == total:
        return "Highly inaccurate"
    else:
        return "Mixed factuality"

cpr_plan = [3]*40 + [4]*60 + [5]*60 + [6]*30 + [7]*10
random.shuffle(cpr_plan)

RESPONSES = []
CLAIMS = []
EVIDENCE = []

claim_counter = 1
evidence_counter = 1

# Generate 200 distinct responses
for i in range(1, 201):
    resp_id = f"L5_{i:03d}"
    topic = TOPICS[(i - 1) % len(TOPICS)]
    style = RESPONSE_STYLES[(i - 1) % len(RESPONSE_STYLES)]
    num_claims = cpr_plan[i - 1]
    
    topic_items = TOPIC_BASE_DATA[topic]
    # Pick a primary item and secondary items for this response
    primary_idx = (i - 1) % len(topic_items)
    selected_items = [topic_items[primary_idx]]
    other_indices = [idx for idx in range(len(topic_items)) if idx != primary_idx]
    random.shuffle(other_indices)
    for idx in other_indices:
        if len(selected_items) < num_claims:
            selected_items.append(topic_items[idx])
            
    # For each selected item, create a distinct claim
    resp_sentences = []
    supports_c = 0
    refutes_c = 0
    nei_c = 0
    halluc_types = []
    
    for c_idx, item in enumerate(selected_items[:num_claims]):
        subject, p1, p2, p3, p4, distractor_p = item
        cid = f"{resp_id}_C{c_idx+1}"
        
        # Decide label to maintain target balance: ~40% SUPPORTS, 35% REFUTES, 25% NEI
        r_roll = random.random()
        if r_roll < 0.40:
            # SUPPORTS
            label = "SUPPORTS"
            diff = random.choice(["easy", "medium", "medium", "hard"])
            htype = "none"
            supports_c += 1
            ctext = f"{subject} is associated with {p1} and {p2}."
            gold_ev = f"{subject} is formally documented in reference materials as relating to {p1} and {p2}, functioning within {p4}."
            partial_ev = f"{subject} has been widely analyzed in studies regarding {p3}."
        elif r_roll < 0.75:
            # REFUTES
            label = "REFUTES"
            diff = random.choice(["medium", "medium", "hard", "easy"])
            # Select hallucination type
            htype_choices = [
                ("wrong_location", f"{subject} is located entirely within Antarctica.", f"{subject} is located in {p2}, not Antarctica."),
                ("wrong_person", f"{subject} was originated by Alexander the Great.", f"{subject} was developed by {p1}, not Alexander the Great."),
                ("wrong_year", f"{subject} was established in the year 1492.", f"{subject} dates to {p3}, not the year 1492."),
                ("negation_error", f"{subject} has never been documented or implemented in {p2}.", f"{subject} has been documented and operates actively in {p2}."),
                ("entity_substitution", f"{subject} is a variety of {distractor_p}.", f"{subject} is an instance of {p4}, not {distractor_p}."),
                ("wrong_number", f"{subject} involves exactly 500 distinct {p1}.", f"{subject} involves {p1} and is classified under {p4}."),
                ("geographic_substitution", f"{subject} is situated on the island of Madagascar.", f"{subject} is based in {p2}, not Madagascar."),
                ("historical_misattribution", f"{subject} was commissioned during the reign of Julius Caesar.", f"{subject} was developed in {p3} by {p1}."),
                ("incorrect_scientific_property", f"{subject} operates exclusively by generating nuclear fission energy.", f"{subject} operates via {p4}, without generating nuclear fission."),
                ("incorrect_technical_specification", f"{subject} is structured as a {distractor_p}.", f"{subject} is structured as a {p4}, not a {distractor_p}.")
            ]
            htype, ctext, gold_ev = random.choice(htype_choices)
            refutes_c += 1
            halluc_types.append(htype)
            partial_ev = f"Scholarly records discuss {subject} in the broader context of {p3}."
        else:
            # NOT_ENOUGH_INFO
            label = "NOT_ENOUGH_INFO"
            diff = random.choice(["hard", "hard", "medium"])
            htype = "none"
            nei_c += 1
            ctext = f"{subject} was officially designated as the supreme standard in {p3}."
            # Evidence talks about subject and p1/p2, but doesn't mention supreme standard designation
            gold_ev = f"{subject} is known for its role in {p4} and historical connection with {p1}."
            partial_ev = f"Historical literature contains various descriptions of {subject}."
            
        resp_sentences.append(ctext)
        
        # Write Gold Evidence
        gold_eid = f"E_{evidence_counter:05d}"
        evidence_counter += 1
        gold_role = "GOLD_SUPPORT" if label == "SUPPORTS" else ("GOLD_REFUTE" if label == "REFUTES" else "NEI_EVIDENCE")
        EVIDENCE.append({
            "evidence_id": gold_eid,
            "claim_id": cid,
            "passage_text": gold_ev,
            "evidence_role": gold_role,
            "relevance_label": "1"
        })
        
        # Write 4 Non-gold candidates
        # 1 Partial
        eid_p = f"E_{evidence_counter:05d}"
        evidence_counter += 1
        EVIDENCE.append({
            "evidence_id": eid_p,
            "claim_id": cid,
            "passage_text": partial_ev,
            "evidence_role": "PARTIAL",
            "relevance_label": "0"
        })
        # 1 Topically related
        eid_t = f"E_{evidence_counter:05d}"
        evidence_counter += 1
        EVIDENCE.append({
            "evidence_id": eid_t,
            "claim_id": cid,
            "passage_text": f"Reference records contain extensive domain knowledge concerning {topic.lower()} principles and historical developments.",
            "evidence_role": "TOPICALLY_RELATED",
            "relevance_label": "0"
        })
        # 2 Irrelevant
        for _ in range(2):
            eid_irr = f"E_{evidence_counter:05d}"
            evidence_counter += 1
            EVIDENCE.append({
                "evidence_id": eid_irr,
                "claim_id": cid,
                "passage_text": random.choice(DISTRACTOR_PASSAGES),
                "evidence_role": "IRRELEVANT",
                "relevance_label": "0"
            })
            
        # Add claim
        claim_type_map = {
            "wrong_location": "location", "wrong_person": "person", "wrong_year": "date",
            "negation_error": "definition", "entity_substitution": "definition",
            "wrong_number": "number", "geographic_substitution": "location",
            "historical_misattribution": "event", "incorrect_scientific_property": "scientific",
            "incorrect_technical_specification": "scientific"
        }
        ctype = claim_type_map.get(htype, random.choice(["definition", "person", "date", "location", "scientific", "organization"]))
        CLAIMS.append({
            "claim_id": cid,
            "response_id": resp_id,
            "claim_text": ctext,
            "claim_type": ctype,
            "label": label,
            "difficulty": diff,
            "hallucination_type": htype,
            "gold_evidence_id": gold_eid
        })
        
    generated_text = " ".join(resp_sentences)
    total_c = len(resp_sentences)
    verdict = get_response_verdict(supports_c, refutes_c, nei_c, total_c)
    f_score = round(supports_c / total_c, 2)
    htypes_str = ";".join(halluc_types) if halluc_types else "none"
    
    RESPONSES.append({
        "response_id": resp_id,
        "generated_text": generated_text,
        "topic": topic,
        "response_style": style,
        "total_claims": str(total_c),
        "supports_count": str(supports_c),
        "refutes_count": str(refutes_c),
        "nei_count": str(nei_c),
        "response_verdict": verdict,
        "factuality_score": f"{f_score:.2f}",
        "hallucination_types": htypes_str,
        "human_verified": "false",
        "source_note": "synthetic_llm_authored"
    })

print(f"Generated {len(RESPONSES)} unique responses, {len(CLAIMS)} claims, {len(EVIDENCE)} evidence rows.")

# Write CSV files
resp_file = os.path.join(OUT_DIR, "heldout_responses.csv")
with open(resp_file, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "response_id", "generated_text", "topic", "response_style", "total_claims",
        "supports_count", "refutes_count", "nei_count", "response_verdict",
        "factuality_score", "hallucination_types", "human_verified", "source_note"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(RESPONSES)
print(f"Wrote {len(RESPONSES)} responses to {resp_file}")

claims_file = os.path.join(OUT_DIR, "heldout_claims.csv")
with open(claims_file, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "claim_id", "response_id", "claim_text", "claim_type", "label",
        "difficulty", "hallucination_type", "gold_evidence_id"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(CLAIMS)
print(f"Wrote {len(CLAIMS)} claims to {claims_file}")

evidence_file = os.path.join(OUT_DIR, "heldout_evidence.csv")
with open(evidence_file, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "evidence_id", "claim_id", "passage_text", "evidence_role", "relevance_label"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(EVIDENCE)
print(f"Wrote {len(EVIDENCE)} evidence rows to {evidence_file}")

print("\nLayer 5 benchmark generation complete.")
