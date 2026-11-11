"""
generate_layer3_claim_extraction.py — Phase 3
Generates:
  1. dataset/layer3_claim_extraction/raw_ai_outputs.jsonl (150 documents)
  2. dataset/layer3_claim_extraction/claim_spans.jsonl (sentence annotations for train)
  3. dataset/layer3_claim_extraction/claim_extraction_test.jsonl (30 test documents annotations)
"""

import os, json

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
OUT_DIR = os.path.join(BASE, "layer3_claim_extraction")
os.makedirs(OUT_DIR, exist_ok=True)

TOPICS = [
    "Technology", "Science", "History", "Geography", "Politics and government",
    "Economics", "Sports", "Medicine and health", "Environment", "Education",
    "Space", "Companies and organizations", "Famous people", "Culture", "General knowledge"
]

PROMPT_TYPES = ["factual_question", "open_explanation", "comparison", "technical", "historical", "conversational"]

# Predefined templates of realistic AI responses with sentence breakdown and annotations
DOC_TEMPLATES = [
    # 1. Technology
    ("Technology", "technical", [
        ("Python is widely regarded as one of the easiest programming languages to learn.", False, None),
        ("Guido van Rossum began developing Python in December 1989 at CWI in the Netherlands.", True, "person"),
        ("The language emphasizes code readability through the use of significant indentation.", True, "definition"),
        ("Many developers prefer Python over C++ for rapid prototyping and data science.", False, None),
        ("Python 3.0 was officially released on December 3, 2008.", True, "date")
    ]),
    ("Technology", "open_explanation", [
        ("The World Wide Web transformed human communication in the late twentieth century.", False, None),
        ("Tim Berners-Lee invented the World Wide Web in 1989 while employed at CERN.", True, "person"),
        ("He developed the first web browser and the Hypertext Transfer Protocol.", True, "scientific"),
        ("This innovation made decentralized document sharing accessible worldwide.", False, None)
    ]),
    ("Technology", "factual_question", [
        ("Linux powers the vast majority of cloud servers and supercomputers today.", False, None),
        ("Linus Torvalds published the first version of the Linux kernel on September 17, 1991.", True, "date"),
        ("The kernel was written in C and assembly language.", True, "scientific"),
        ("Linux is released under the GNU General Public License version 2.", True, "organization")
    ]),
    ("Technology", "comparison", [
        ("Solid-state drives offer substantially faster data transfer rates than mechanical hard drives.", False, None),
        ("SSDs utilize NAND flash memory to store digital information non-volatilely.", True, "scientific"),
        ("Fujio Masuoka invented flash memory while working at Toshiba in the 1980s.", True, "person"),
        ("Users seeking maximum responsiveness should always choose an SSD.", False, None)
    ]),
    ("Technology", "historical", [
        ("The development of the transistor marked the beginning of modern microelectronics.", False, None),
        ("John Bardeen, Walter Brattain, and William Shockley invented the transistor at Bell Labs in 1947.", True, "person"),
        ("They received the Nobel Prize in Physics in 1956 for their discovery.", True, "event"),
        ("Silicon eventually replaced germanium as the primary semiconductor material.", True, "scientific")
    ]),
    ("Technology", "conversational", [
        ("Git is indispensable for modern collaborative software engineering.", False, None),
        ("Linus Torvalds designed Git in 2005 to manage Linux kernel development.", True, "person"),
        ("Git uses cryptographic hashing with SHA-1 to maintain historical data integrity.", True, "technical_specification" if False else "scientific"),
        ("Learning Git commands takes time but is well worth the effort.", False, None)
    ]),
    ("Technology", "technical", [
        ("Relational databases remain the backbone of enterprise data infrastructure.", False, None),
        ("Edgar F. Codd introduced the relational model in a seminal research paper in 1970.", True, "date"),
        ("Structured Query Language was subsequently developed at IBM to manipulate relational data.", True, "organization"),
        ("ACID properties ensure reliable processing of database transactions.", True, "definition")
    ]),
    ("Technology", "open_explanation", [
        ("Public key cryptography solved the problem of secure key exchange across insecure channels.", False, None),
        ("Whitfield Diffie and Martin Hellman introduced public-key cryptography principles in 1976.", True, "person"),
        ("The RSA algorithm was developed in 1977 by Ron Rivest, Adi Shamir, and Leonard Adleman.", True, "person"),
        ("Cryptographic security depends on the computational hardness of mathematical problems.", True, "scientific")
    ]),
    ("Technology", "factual_question", [
        ("Optical fiber networks carry the majority of global internet traffic today.", False, None),
        ("Charles K. Kao demonstrated that high-purity silica glass could transmit light over long distances in 1966.", True, "person"),
        ("Total internal reflection allows light pulses to travel along the fiber core.", True, "scientific"),
        ("Kao was awarded the 2009 Nobel Prize in Physics for this work.", True, "event")
    ]),
    ("Technology", "comparison", [
        ("JavaScript and TypeScript have distinct tradeoffs in large web application codebases.", False, None),
        ("Brendan Eich created JavaScript in 1995 while working at Netscape.", True, "person"),
        ("TypeScript was developed by Microsoft and first released in October 2012.", True, "organization"),
        ("Static type checking helps identify common runtime errors early in development.", False, None)
    ]),

    # 2. Science
    ("Science", "factual_question", [
        ("The speed of light in vacuum is exactly 299,792,458 metres per second.", True, "number"),
        ("This physical constant is denoted by the letter c in relativistic equations.", True, "definition"),
        ("No physical object with mass can accelerate to or exceed this speed in spacetime.", True, "scientific"),
        ("Understanding the speed of light fundamentally reshaped twentieth-century physics.", False, None)
    ]),
    ("Science", "open_explanation", [
        ("Photosynthesis is one of the most vital biochemical processes on planet Earth.", False, None),
        ("Plants convert carbon dioxide and water into glucose and oxygen using sunlight.", True, "scientific"),
        ("Chlorophyll pigments inside chloroplasts absorb photons primarily in blue and red wavelengths.", True, "scientific"),
        ("Without photosynthesis, aerobic organisms could not survive.", False, None)
    ]),
    ("Science", "historical", [
        ("The discovery of DNA's double helix was a milestone in biological science.", False, None),
        ("James Watson and Francis Crick proposed the double helix structure of DNA in 1953.", True, "person"),
        ("Rosalind Franklin produced the crucial X-ray diffraction images of DNA crystals.", True, "person"),
        ("The Nobel Prize in Physiology or Medicine was awarded in 1962 for this achievement.", True, "date")
    ]),
    ("Science", "technical", [
        ("Mitochondria are often called the powerhouses of eukaryotic cells.", False, None),
        ("Mitochondria generate adenosine triphosphate through the process of oxidative phosphorylation.", True, "scientific"),
        ("They contain their own distinct circular DNA inherited maternally in humans.", True, "scientific"),
        ("Mitochondrial dysfunction is linked to numerous metabolic and neurological disorders.", False, None)
    ]),
    ("Science", "comparison", [
        ("Nuclear fission and fusion are fundamentally opposite nuclear reactions.", False, None),
        ("Fission splits a heavy atomic nucleus into lighter nuclei with energy release.", True, "definition"),
        ("Nuclear fusion combines light atomic nuclei to form a heavier nucleus.", True, "definition"),
        ("The Sun generates energy primarily through nuclear fusion of hydrogen into helium.", True, "scientific")
    ]),
    ("Science", "factual_question", [
        ("Dmitri Mendeleev arranged the periodic table according to atomic weights in 1869.", True, "person"),
        ("The modern periodic table is ordered strictly by increasing atomic number.", True, "scientific"),
        ("Henry Moseley established the concept of atomic number using X-ray spectroscopy in 1913.", True, "date"),
        ("The table currently includes 118 confirmed chemical elements.", True, "number")
    ]),
    ("Science", "open_explanation", [
        ("Gravitational waves are ripples in the fabric of spacetime caused by accelerated masses.", True, "scientific"),
        ("Albert Einstein predicted the existence of gravitational waves in 1916.", True, "date"),
        ("The LIGO interferometers first detected gravitational waves directly on September 14, 2015.", True, "event"),
        ("This detection opened an exciting new observational window into the universe.", False, None)
    ]),
    ("Science", "technical", [
        ("CRISPR-Cas9 provides precise molecular scissors for targeted genome editing.", False, None),
        ("Emmanuelle Charpentier and Jennifer Doudna published the CRISPR-Cas9 genome editing mechanism in 2012.", True, "person"),
        ("The Cas9 enzyme uses guide RNA sequences to locate and cleave target DNA.", True, "scientific"),
        ("They were awarded the 2020 Nobel Prize in Chemistry.", True, "event")
    ]),
    ("Science", "conversational", [
        ("Water possesses unusual physical and chemical properties compared to similar molecules.", False, None),
        ("A single water molecule consists of two hydrogen atoms covalently bonded to one oxygen atom.", True, "definition"),
        ("Hydrogen bonding causes liquid water to be denser than solid ice at standard pressure.", True, "scientific"),
        ("These characteristics make water an exceptional universal solvent.", False, None)
    ]),
    ("Science", "historical", [
        ("The discovery of radioactivity challenged classical views of immutable atomic structures.", False, None),
        ("Henri Becquerel discovered radioactivity in uranium salts in 1896.", True, "person"),
        ("Marie Curie and Pierre Curie isolated the radioactive elements polonium and radium in 1898.", True, "date"),
        ("Radioactivity plays crucial roles in both modern energy and medical treatments.", False, None)
    ]),

    # 3. History
    ("History", "factual_question", [
        ("Apollo 11 was the first spaceflight that successfully landed humans on the Moon.", True, "event"),
        ("Neil Armstrong and Buzz Aldrin landed the lunar module Eagle on July 20, 1969.", True, "person"),
        ("Michael Collins remained in lunar orbit aboard the command module Columbia.", True, "person"),
        ("The mission fulfilled President John F. Kennedy's goal of reaching the Moon within the decade.", False, None)
    ]),
    ("History", "open_explanation", [
        ("The French Revolution drastically altered the political landscape of Europe.", False, None),
        ("The storming of the Bastille on July 14, 1789 symbolizes the outbreak of the revolution.", True, "date"),
        ("The National Assembly adopted the Declaration of the Rights of Man and of the Citizen in August 1789.", True, "event"),
        ("King Louis XVI was executed by guillotine in January 1793.", True, "person")
    ]),
    ("History", "historical", [
        ("The Magna Carta established foundational principles regarding legal limits on royal power.", False, None),
        ("King John of England sealed the Magna Carta at Runnymede in June 1215.", True, "person"),
        ("Clause 39 guaranteed that no free man could be imprisoned without lawful judgment.", True, "definition"),
        ("This document inspired future constitutional documents including the US Bill of Rights.", False, None)
    ]),
    ("History", "comparison", [
        ("The construction of major shipping canals transformed nineteenth-century maritime commerce.", False, None),
        ("The Suez Canal opened in Egypt in November 1869, connecting the Mediterranean and Red Seas.", True, "location"),
        ("The Panama Canal opened in August 1914, connecting the Atlantic and Pacific Oceans.", True, "date"),
        ("Both engineering feats drastically reduced oceanic voyage times.", False, None)
    ]),
    ("History", "factual_question", [
        ("The Berlin Wall divided the city of Berlin during the Cold War.", True, "location"),
        ("East German authorities began building the wall in August 1961.", True, "date"),
        ("The Berlin Wall was opened on November 9, 1989 following widespread civil protests.", True, "date"),
        ("Its collapse symbolized the end of European communist authoritarianism.", False, None)
    ]),
    ("History", "open_explanation", [
        ("The Industrial Revolution marked a major transition in human productive capacity.", False, None),
        ("It began in Great Britain during the mid-eighteenth century with mechanized textile spinning.", True, "location"),
        ("James Watt patented an improved steam engine design in 1769.", True, "person"),
        ("Urbanization accelerated dramatically as factory jobs multiplied.", False, None)
    ]),
    ("History", "conversational", [
        ("Johannes Gutenberg revolutionized the distribution of knowledge in Europe.", False, None),
        ("Gutenberg developed movable metal type printing in Mainz around 1440.", True, "location"),
        ("The Gutenberg Bible was printed between 1450 and 1455.", True, "date"),
        ("Mass printing accelerated both the Renaissance and the Scientific Revolution.", False, None)
    ]),
    ("History", "historical", [
        ("The United Nations was created in the aftermath of the Second World War.", False, None),
        ("Representatives from 50 nations met at the San Francisco Conference in 1945.", True, "number"),
        ("The UN Charter came into official force on October 24, 1945.", True, "date"),
        ("The organization seeks to prevent international conflicts and foster global cooperation.", False, None)
    ]),
    ("History", "factual_question", [
        ("The ancient Olympic Games were held in Olympia, Greece.", True, "location"),
        ("The first recorded Olympic Games took place in 776 BC.", True, "date"),
        ("Competitions were held in honour of the Greek god Zeus every four years.", True, "definition"),
        ("Athletic excellence was deeply intertwined with religious and civic virtue.", False, None)
    ]),
    ("History", "comparison", [
        ("World War I and World War II were the most devastating global conflicts in modern history.", False, None),
        ("The Treaty of Versailles was signed on June 28, 1919 to end World War I hostilities.", True, "date"),
        ("World War II concluded in 1945 following the unconditional surrender of Axis forces.", True, "date"),
        ("The human and economic toll reshaped the geopolitical world order.", False, None)
    ]),
]

# Generate additional variations across remaining topics to hit 150 train documents and 30 test documents
import random
random.seed(123)

ALL_DOCS = []
doc_id = 1

# Add templates
for topic, ptype, sents in DOC_TEMPLATES:
    full_text = " ".join(s[0] for s in sents)
    ALL_DOCS.append({
        "doc_id": f"CE_{doc_id:03d}",
        "topic": topic,
        "prompt_type": ptype,
        "generated_text": full_text,
        "sentences": sents,
        "human_verified": False,
        "source_note": "synthetic_llm_authored"
    })
    doc_id += 1

# Generate remaining to reach 150 docs across 15 topics (10 per topic)
TOPIC_FACT_SETS = {
    "Geography": [
        ("Mount Everest stands at an official elevation of 8,848.86 metres.", True, "number", "Mountains"),
        ("The mountain is situated on the international border between Nepal and China.", True, "location", "Mountains"),
        ("Everest is widely considered the ultimate mountaineering challenge on Earth.", False, None, "Mountains"),
        ("The Amazon River is the largest river by discharge volume of water worldwide.", True, "comparison", "Rivers"),
        ("The Nile River flows for approximately 6,650 kilometres through northeastern Africa.", True, "number", "Rivers"),
        ("The Sahara Desert covers more than 9 million square kilometres of North Africa.", True, "number", "Deserts"),
        ("The Mariana Trench reaches its deepest point at the Challenger Deep at nearly 11,000 metres.", True, "location", "Oceans"),
        ("Lake Baikal holds approximately 20 percent of Earth's unfrozen surface fresh water.", True, "number", "Lakes"),
        ("Lake Baikal is located in southern Siberia, Russia.", True, "location", "Lakes"),
        ("The Grand Canyon was carved by the Colorado River over millions of years.", True, "cause_effect", "Landforms")
    ],
    "Politics and government": [
        ("A bicameral legislature consists of two separate deliberative assemblies.", True, "definition", "Legislatures"),
        ("The United States Congress comprises the Senate and the House of Representatives.", True, "organization", "Legislatures"),
        ("Effective governance requires a delicate balance of competing political interests.", False, None, "Legislatures"),
        ("The Universal Declaration of Human Rights was adopted by the UN General Assembly in 1948.", True, "date", "Human Rights"),
        ("The document proclaims thirty fundamental rights and freedoms belonging to all humans.", True, "number", "Human Rights"),
        ("The International Court of Justice has its permanent headquarters in The Hague, Netherlands.", True, "location", "Courts"),
        ("The US Constitution was signed in Philadelphia in September 1787.", True, "location", "Constitutions"),
        ("Judicial review allows courts to determine the constitutionality of legislative statutes.", True, "definition", "Judiciary"),
        ("Federalism divides governing authority between national and regional state entities.", True, "definition", "Federalism"),
        ("The European Parliament meets in Strasbourg and Brussels.", True, "location", "EU")
    ],
    "Economics": [
        ("Gross Domestic Product measures the total monetary value of final goods produced within a country.", True, "definition", "Macroeconomics"),
        ("Economists frequently debate the best indicators for assessing long-term national prosperity.", False, None, "Macroeconomics"),
        ("Central banks use policy interest rates to influence inflation and macroeconomic activity.", True, "cause_effect", "Monetary"),
        ("The Federal Reserve was established by the US Congress in December 1913.", True, "date", "Central Banks"),
        ("The Bretton Woods Conference in 1944 established the International Monetary Fund.", True, "organization", "Institutions"),
        ("David Ricardo introduced the theory of comparative advantage in 1817.", True, "person", "Trade"),
        ("Inflation represents the sustained general increase in price levels over time.", True, "definition", "Inflation"),
        ("Price elasticity of demand measures consumer responsiveness to price alterations.", True, "definition", "Microeconomics"),
        ("The Gini coefficient is a statistical index measuring income or wealth distribution inequality.", True, "definition", "Inequality"),
        ("Tariffs are taxes imposed on imported merchandise by domestic authorities.", True, "definition", "Trade")
    ],
    "Sports": [
        ("The first modern Olympic Games took place in Athens, Greece in April 1896.", True, "event", "Olympics"),
        ("Athletic competition inspires millions of sports fans around the globe.", False, None, "Olympics"),
        ("The standard marathon distance is 42.195 kilometres.", True, "number", "Track"),
        ("James Naismith invented basketball in Springfield, Massachusetts in December 1891.", True, "person", "Basketball"),
        ("The inaugural FIFA World Cup was hosted and won by Uruguay in 1930.", True, "event", "Soccer"),
        ("Wimbledon is the oldest tennis tournament in the world, founded in 1877.", True, "date", "Tennis"),
        ("Usain Bolt set the world 100-metre record of 9.58 seconds in Berlin in 2009.", True, "number", "Track"),
        ("Formula One held its first championship Grand Prix at Silverstone in 1950.", True, "location", "Motorsport"),
        ("The Stanley Cup was originally commissioned in 1892 as an amateur challenge cup.", True, "date", "Hockey"),
        ("Super Bowl I was contested between Green Bay and Kansas City in January 1967.", True, "event", "Football")
    ],
    "Medicine and health": [
        ("Alexander Fleming discovered penicillin at St Mary's Hospital in London in 1928.", True, "person", "Antibiotics"),
        ("The development of antibiotics ranks among humanity's greatest medical achievements.", False, None, "Antibiotics"),
        ("The World Health Organization declared smallpox eradicated in May 1980.", True, "event", "Eradication"),
        ("Frederick Banting and Charles Best isolated insulin in Toronto in 1921.", True, "person", "Insulin"),
        ("Jonas Salk announced his inactivated polio vaccine as effective in April 1955.", True, "date", "Vaccines"),
        ("The human genome contains roughly 3 billion base pairs of DNA.", True, "number", "Genetics"),
        ("Type 1 diabetes is an autoimmune condition resulting in destruction of insulin-producing beta cells.", True, "definition", "Diabetes"),
        ("Hemoglobin is an iron-rich protein in red blood cells that transports oxygen.", True, "definition", "Hematology"),
        ("Wilhelm Rontgen discovered X-rays in 1895 and won the 1901 Nobel Prize in Physics.", True, "person", "Imaging"),
        ("Regular cardiovascular exercise provides substantial health benefits for all individuals.", False, None, "Wellness")
    ],
    "Environment": [
        ("The Paris Agreement was adopted by 196 parties at COP 21 in December 2015.", True, "event", "Climate"),
        ("Combating global environmental degradation requires unprecedented international cooperation.", False, None, "Climate"),
        ("The Montreal Protocol of 1987 phased out ozone-depleting chlorofluorocarbons.", True, "event", "Ozone"),
        ("The Amazon rainforest absorbs and stores billions of tonnes of carbon dioxide annually.", True, "scientific", "Forests"),
        ("Ocean acidification results from marine absorption of excess atmospheric carbon dioxide.", True, "cause_effect", "Oceans"),
        ("The IPCC was founded in 1988 by the WMO and UNEP.", True, "organization", "IPCC"),
        ("Coral bleaching occurs when corals expel symbiotic algae due to thermal stress.", True, "cause_effect", "Reefs"),
        ("Wetlands serve as crucial natural filtration systems for freshwater watersheds.", True, "scientific", "Ecosystems"),
        ("Biodiversity loss threatens the stability of global agricultural and marine ecosystems.", False, None, "Ecosystems"),
        ("The Ramsar Convention protects designated wetlands of international importance.", True, "organization", "Wetlands")
    ],
    "Education": [
        ("Formative assessment provides ongoing diagnostic feedback during the learning process.", True, "definition", "Assessment"),
        ("Summative assessment evaluates student achievement at the conclusion of an instructional period.", True, "definition", "Assessment"),
        ("Carefully designed evaluations enhance both teacher efficacy and student motivation.", False, None, "Assessment"),
        ("Benjamin Bloom published the taxonomy of educational objectives in 1956.", True, "person", "Taxonomy"),
        ("Maria Montessori founded her first Casa dei Bambini school in Rome in 1907.", True, "location", "Montessori"),
        ("The University of Bologna was established in Italy in 1088.", True, "date", "Universities"),
        ("Lev Vygotsky formulated the concept of the Zone of Proximal Development in developmental psychology.", True, "person", "Psychology"),
        ("The OECD administers the PISA assessment to 15-year-old students internationally every three years.", True, "organization", "PISA"),
        ("Jean Piaget proposed four successive stages in child cognitive development.", True, "number", "Piaget"),
        ("Collaborative learning approaches often yield superior engagement in classroom settings.", False, None, "Pedagogy")
    ],
    "Space": [
        ("Mars has two small natural satellites named Phobos and Deimos.", True, "number", "Mars"),
        ("Asaph Hall discovered both Martian moons in August 1877.", True, "person", "Mars"),
        ("Exploring Mars represents the next logical leap for human interplanetary exploration.", False, None, "Mars"),
        ("The James Webb Space Telescope was launched into space on December 25, 2021.", True, "date", "Telescopes"),
        ("Voyager 1 entered interstellar space in August 2012.", True, "date", "Probes"),
        ("The International Astronomical Union reclassified Pluto as a dwarf planet in 2006.", True, "organization", "Pluto"),
        ("Sagittarius A* is a supermassive black hole located at the centre of the Milky Way.", True, "scientific", "Black Holes"),
        ("Jupiter's Great Red Spot is a persistent anticyclonic storm larger than planet Earth.", True, "comparison", "Jupiter"),
        ("Saturn's planetary rings consist predominantly of water ice particles and dust.", True, "scientific", "Saturn"),
        ("Apollo 17 was NASA's final crewed lunar landing mission in December 1972.", True, "event", "Apollo")
    ],
    "Companies and organizations": [
        ("Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.", True, "person", "Tech Companies"),
        ("Microsoft was founded by Bill Gates and Paul Allen in Albuquerque in April 1975.", True, "person", "Tech Companies"),
        ("Technological entrepreneurship drives significant modern economic transformation.", False, None, "Tech Companies"),
        ("Google was incorporated in September 1998 by Larry Page and Sergey Brin.", True, "date", "Tech Companies"),
        ("Jeff Bezos founded Amazon in July 1994 as an online bookstore.", True, "person", "Tech Companies"),
        ("Henry Dunant founded the International Committee of the Red Cross in Geneva in 1863.", True, "person", "NGOs"),
        ("CERN operates the Large Hadron Collider particle accelerator near Geneva, Switzerland.", True, "location", "CERN"),
        ("Amnesty International was established in London by Peter Benenson in July 1961.", True, "date", "NGOs"),
        ("The World Trade Organization succeeded the GATT agreement on January 1, 1995.", True, "organization", "WTO"),
        ("Elon Musk founded SpaceX in March 2002 with headquarters in Hawthorne, California.", True, "location", "SpaceX")
    ],
    "Famous people": [
        ("Albert Einstein won the 1921 Nobel Prize in Physics for the photoelectric effect.", True, "event", "Physicists"),
        ("Einstein remains one of the most celebrated intellectual figures in human history.", False, None, "Physicists"),
        ("Marie Curie received Nobel Prizes in both Physics in 1903 and Chemistry in 1911.", True, "number", "Scientists"),
        ("Sir Isaac Newton published his laws of motion and gravitation in 1687.", True, "date", "Physicists"),
        ("Nelson Mandela served as President of South Africa from 1994 to 1999.", True, "date", "Leaders"),
        ("Charles Darwin published On the Origin of Species in November 1859.", True, "date", "Biologists"),
        ("Ada Lovelace published the first computer algorithm in 1843 for Babbage's Analytical Engine.", True, "person", "Computing"),
        ("Leonardo da Vinci painted both the Mona Lisa and The Last Supper.", True, "relationship", "Artists"),
        ("Martin Luther King Jr. delivered his 'I Have a Dream' speech in Washington in August 1963.", True, "location", "Civil Rights"),
        ("Alan Turing developed key cryptographic machines at Bletchley Park during World War II.", True, "location", "Turing")
    ],
    "Culture": [
        ("A traditional Japanese haiku poem adheres to a strict 5-7-5 syllable structure.", True, "definition", "Poetry"),
        ("Poetic expression allows writers to capture profound human emotions with brevity.", False, None, "Poetry"),
        ("Gothic architecture emerged in twelfth-century France with the Abbey of Saint-Denis.", True, "location", "Architecture"),
        ("Kabuki is a classical Japanese theatre form that originated in Kyoto in the seventeenth century.", True, "location", "Theatre"),
        ("Tango dance developed in the late nineteenth century in Buenos Aires and Montevideo.", True, "location", "Dance"),
        ("The Shakespearean sonnet contains 14 lines in iambic pentameter with an ABAB CDCD EFEF GG rhyme scheme.", True, "number", "Poetry"),
        ("Impressionist painting originated in Paris in the 1870s with artists such as Claude Monet.", True, "date", "Art"),
        ("Flamenco music and dance is native to the Andalusia region in southern Spain.", True, "location", "Music"),
        ("Jazz developed in the African-American communities of New Orleans around the turn of the twentieth century.", True, "location", "Music"),
        ("The Lumiere brothers held the first public commercial cinema screening in Paris in December 1895.", True, "date", "Cinema")
    ],
    "General knowledge": [
        ("Pure water boils at 100 degrees Celsius and freezes at 0 degrees Celsius at 1 atmosphere of pressure.", True, "number", "Water"),
        ("Basic scientific literacy helps individuals make well-informed everyday decisions.", False, None, "General"),
        ("Earth rotates on its axis relative to distant stars in approximately 23 hours and 56 minutes.", True, "number", "Earth"),
        ("A standard leap year in the Gregorian calendar contains 366 days instead of 365.", True, "number", "Calendar"),
        ("The English alphabet consists of 26 letters including five vowel characters.", True, "number", "Language"),
        ("A prime number is an integer greater than 1 with no positive divisors other than 1 and itself.", True, "definition", "Math"),
        ("Earth's atmosphere comprises approximately 78 percent nitrogen and 21 percent oxygen by volume.", True, "number", "Atmosphere"),
        ("The International Date Line roughly follows the 180th meridian across the Pacific Ocean.", True, "location", "Geography"),
        ("Diamond rates as the maximum hardness of 10 on the Mohs mineral scale.", True, "number", "Minerals"),
        ("The speed of sound in dry air at 20 degrees Celsius is approximately 343 metres per second.", True, "number", "Sound")
    ]
}

# Assemble remaining documents up to 150
while len(ALL_DOCS) < 150:
    for topic, facts in TOPIC_FACT_SETS.items():
        if len(ALL_DOCS) >= 150:
            break
        # Pick 3-5 sentences
        sample_facts = random.sample(facts, min(len(facts), random.randint(3, 5)))
        sents = [(f[0], f[1], f[2]) for f in sample_facts]
        ptype = random.choice(PROMPT_TYPES)
        full_text = " ".join(s[0] for s in sents)
        ALL_DOCS.append({
            "doc_id": f"CE_{doc_id:03d}",
            "topic": topic,
            "prompt_type": ptype,
            "generated_text": full_text,
            "sentences": sents,
            "human_verified": False,
            "source_note": "synthetic_llm_authored"
        })
        doc_id += 1

print(f"Generated {len(ALL_DOCS)} training documents.")

# Generate 30 held-out test documents
TEST_DOCS = []
test_doc_id = 1
for i in range(30):
    topic = TOPICS[i % len(TOPICS)]
    facts = TOPIC_FACT_SETS.get(topic, TOPIC_FACT_SETS["General knowledge"])
    sample_facts = random.sample(facts, random.randint(3, 4))
    sents = [(f[0], f[1], f[2]) for f in sample_facts]
    ptype = random.choice(PROMPT_TYPES)
    full_text = " ".join(s[0] for s in sents)
    TEST_DOCS.append({
        "doc_id": f"CE_TEST_{test_doc_id:03d}",
        "topic": topic,
        "prompt_type": ptype,
        "generated_text": full_text,
        "sentences": sents,
        "human_verified": False,
        "source_note": "synthetic_llm_authored"
    })
    test_doc_id += 1

print(f"Generated {len(TEST_DOCS)} test documents.")

# ── Write Files ──────────────────────────────────────────────────────────────
# 1. raw_ai_outputs.jsonl
raw_path = os.path.join(OUT_DIR, "raw_ai_outputs.jsonl")
with open(raw_path, "w", encoding="utf-8") as f:
    for d in ALL_DOCS:
        out_entry = {
            "doc_id": d["doc_id"],
            "topic": d["topic"],
            "prompt_type": d["prompt_type"],
            "generated_text": d["generated_text"],
            "human_verified": d["human_verified"],
            "source_note": d["source_note"]
        }
        f.write(json.dumps(out_entry, ensure_ascii=True) + "\n")
print(f"Wrote {len(ALL_DOCS)} raw docs to {raw_path}")

# 2. claim_spans.jsonl
spans_path = os.path.join(OUT_DIR, "claim_spans.jsonl")
total_spans = 0
with open(spans_path, "w", encoding="utf-8") as f:
    for d in ALL_DOCS:
        for idx, (stext, is_claim, ctype) in enumerate(d["sentences"]):
            span = [0, len(stext)] if is_claim else None
            entry = {
                "doc_id": d["doc_id"],
                "sentence_id": idx,
                "sentence_text": stext,
                "is_factual_claim": is_claim,
                "claim_type": ctype,
                "claim_span": span
            }
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
            total_spans += 1
print(f"Wrote {total_spans} sentence annotations to {spans_path}")

# 3. claim_extraction_test.jsonl
test_spans_path = os.path.join(OUT_DIR, "claim_extraction_test.jsonl")
total_test_spans = 0
with open(test_spans_path, "w", encoding="utf-8") as f:
    for d in TEST_DOCS:
        for idx, (stext, is_claim, ctype) in enumerate(d["sentences"]):
            span = [0, len(stext)] if is_claim else None
            entry = {
                "doc_id": d["doc_id"],
                "sentence_id": idx,
                "sentence_text": stext,
                "is_factual_claim": is_claim,
                "claim_type": ctype,
                "claim_span": span
            }
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
            total_test_spans += 1
print(f"Wrote {total_test_spans} test sentence annotations to {test_spans_path}")

print("\nLayer 3 Claim Extraction dataset generation complete.")
