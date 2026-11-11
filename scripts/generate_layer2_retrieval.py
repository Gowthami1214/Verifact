"""
generate_layer2_retrieval.py — Phase 2
Generates:
  1. dataset/layer2_retrieval/passage_pool.jsonl (~300+ factual passages)
  2. dataset/layer2_retrieval/claim_candidates.csv (5 candidates per Layer 1 train claim)
  3. dataset/layer2_retrieval/retrieval_test.csv (50 claims x 10 candidate passages)
"""

import os, csv, json, random

random.seed(42)

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
OUT_DIR = os.path.join(BASE, "layer2_retrieval")
os.makedirs(OUT_DIR, exist_ok=True)

TOPIC_PASSAGES = {
    "Technology": [
        ("Python Programming Language", "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Its design emphasizes code readability and simplicity.", "Wikipedia"),
        ("World Wide Web Invention", "The World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN in Geneva, Switzerland.", "Wikipedia"),
        ("Linux Operating System", "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, first released by Linus Torvalds in September 1991.", "Wikipedia"),
        ("Transistor Invention", "The point-contact transistor was invented in December 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs.", "scientific"),
        ("Relational Database Management", "The relational model for database management was proposed by Edgar F. Codd in 1970 while working at IBM Research.", "scientific"),
        ("JavaScript Language", "JavaScript was created by Brendan Eich in 1995 while working at Netscape Communications Corporation.", "Wikipedia"),
        ("TCP/IP Protocol", "The Transmission Control Protocol and Internet Protocol (TCP/IP) suite was designed by Vint Cerf and Bob Kahn in the 1970s.", "scientific"),
        ("Quantum Computing Bits", "Quantum computers use qubits which can exist in superposition states, unlike classical bits which represent either 0 or 1.", "scientific"),
        ("Optical Fiber Communication", "Optical fibers transmit data as light pulses through strands of glass or plastic, enabling high-speed long-distance telecommunications.", "scientific"),
        ("Git Version Control", "Git is a distributed version control system created by Linus Torvalds in 2005 for development of the Linux kernel.", "Wikipedia"),
        ("C Programming Language", "The C programming language was developed by Dennis Ritchie between 1969 and 1973 at Bell Labs.", "Wikipedia"),
        ("Artificial Neural Networks", "Artificial neural networks are computing systems inspired by biological neural networks that constitute animal brains.", "scientific"),
        ("Machine Learning Supervised", "Supervised learning algorithms build a mathematical model of a set of data that contains both inputs and desired outputs.", "scientific"),
        ("Flash Memory", "Flash memory is an electronic non-volatile computer storage medium invented by Fujio Masuoka at Toshiba in the early 1980s.", "scientific"),
        ("Ethernet Networking", "Ethernet was developed at Xerox PARC by Robert Metcalfe and others in the early 1970s for local area networking.", "Wikipedia"),
        ("Solid State Drives", "Solid-state drives use integrated circuit assemblies to store data persistently, typically using flash memory.", "scientific"),
        ("Bluetooth Technology", "Bluetooth is a short-range wireless technology standard managed by the Bluetooth Special Interest Group, introduced in 1998.", "official_organization"),
        ("RISC-V Architecture", "RISC-V is an open standard instruction set architecture based on established reduced instruction set computer principles.", "scientific"),
        ("Graphics Processing Units", "GPUs are specialized electronic circuits designed to accelerate computer graphics processing and parallel workloads.", "scientific"),
        ("Public Key Cryptography", "Public-key cryptography was introduced publicly by Whitfield Diffie and Martin Hellman in their landmark 1976 paper.", "scientific"),
        ("Android Operating System", "Android was developed by Android Inc., which Google acquired in 2005, and is based on a modified version of the Linux kernel.", "Wikipedia"),
    ],
    "Science": [
        ("Speed of Light", "The speed of light in vacuum is exactly 299,792,458 metres per second, a fundamental physical constant.", "scientific"),
        ("DNA Double Helix", "The double helix structure of DNA was determined in 1953 by James Watson and Francis Crick, using X-ray diffraction data from Rosalind Franklin.", "scientific"),
        ("General Relativity", "Albert Einstein published the general theory of relativity in 1915, describing gravity as the curvature of spacetime.", "scientific"),
        ("Periodic Table", "Dmitri Mendeleev formulated the periodic law and published the first widely recognized periodic table in 1869.", "scientific"),
        ("Photosynthesis Process", "Photosynthesis is used by plants and other organisms to convert light energy into chemical energy stored in carbohydrate molecules.", "scientific"),
        ("Water Chemical Formula", "Water is a chemical compound consisting of two hydrogen atoms bonded to one oxygen atom, denoted as H2O.", "scientific"),
        ("Hubble Space Telescope", "The Hubble Space Telescope was launched into low Earth orbit in April 1990 aboard the Space Shuttle Discovery.", "scientific"),
        ("Electron Discovery", "The electron was discovered in 1897 by J. J. Thomson during experiments with cathode ray tubes.", "scientific"),
        ("Neutron Discovery", "James Chadwick discovered the neutron in 1932, for which he received the Nobel Prize in Physics in 1935.", "scientific"),
        ("Plate Tectonics Theory", "Plate tectonics is the scientific theory describing the large-scale motion of seven large plates and many smaller plates of Earth's lithosphere.", "scientific"),
        ("First Law of Thermodynamics", "The first law of thermodynamics states that energy can neither be created nor destroyed, only transformed from one form to another.", "scientific"),
        ("Mitochondria Function", "Mitochondria are double-membrane-bound organelles found in most eukaryotic organisms that generate most of the cell's ATP.", "scientific"),
        ("Big Bang Theory", "The Big Bang model suggests the universe expanded from an extremely high-density and high-temperature initial state roughly 13.8 billion years ago.", "scientific"),
        ("Bohr Atomic Model", "Niels Bohr proposed his atomic model in 1913, introducing quantized electron orbits around the atomic nucleus.", "scientific"),
        ("Cell Theory", "Cell theory states that all living organisms are composed of cells and that new cells arise from pre-existing cells.", "scientific"),
        ("Radioactivity Discovery", "Henri Becquerel discovered radioactivity in 1896 while investigating phosphorescence in uranium salts.", "scientific"),
        ("CRISPR Gene Editing", "CRISPR-Cas9 is a technology that allows geneticists and medical researchers to edit parts of the genome by cutting DNA sequences.", "scientific"),
        ("Ozone Layer Function", "The ozone layer absorbs between 97 and 99 percent of the Sun's medium-frequency ultraviolet light in Earth's stratosphere.", "scientific"),
        ("Penicillin Antibiotic", "Alexander Fleming discovered penicillin in 1928 when he noticed that a Penicillium mould inhibited bacterial growth.", "scientific"),
        ("Gravitational Waves", "Gravitational waves were directly detected for the first time on September 14, 2015 by the LIGO interferometers.", "scientific"),
        ("Higgs Boson", "The Higgs boson was discovered at the Large Hadron Collider at CERN in 2012, confirming the existence of the Higgs field.", "scientific"),
    ],
    "History": [
        ("Apollo 11 Moon Landing", "Apollo 11 was the first crewed mission to land on the Moon on July 20, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins.", "Wikipedia"),
        ("French Revolution", "The French Revolution began in 1789 with the storming of the Bastille and led to the abolition of the French monarchy.", "Wikipedia"),
        ("Magna Carta", "The Magna Carta was signed in June 1215 at Runnymede by King John of England, establishing that monarchial rule was subject to law.", "Wikipedia"),
        ("Fall of the Berlin Wall", "The Berlin Wall fell on November 9, 1989, symbolizing the collapse of Soviet-backed communist regimes in Eastern Europe.", "Wikipedia"),
        ("Treaty of Versailles", "The Treaty of Versailles was signed on June 28, 1919, officially ending World War I between Germany and the Allied Powers.", "Wikipedia"),
        ("United Nations Founding", "The United Nations was founded on October 24, 1945 following the ratification of the UN Charter by the five permanent Security Council members.", "official_organization"),
        ("Industrial Revolution", "The Industrial Revolution began in Great Britain in the mid-18th century, transitioning manufacturing from hand production to machines.", "Wikipedia"),
        ("Emancipation Proclamation", "President Abraham Lincoln issued the Emancipation Proclamation on January 1, 1863, declaring all enslaved persons in Confederate states free.", "government"),
        ("Ancient Olympic Games", "The ancient Olympic Games originated in Olympia, Greece, with the first recorded games held in 776 BC.", "Wikipedia"),
        ("Battle of Hastings", "The Battle of Hastings took place on October 14, 1066, resulting in the Norman conquest of England under William the Conqueror.", "Wikipedia"),
        ("Roman Empire Fall", "The Western Roman Empire fell in 476 AD when the last Western Roman Emperor, Romulus Augustulus, was deposed by Odoacer.", "Wikipedia"),
        ("Suez Canal Opening", "The Suez Canal in Egypt officially opened on November 17, 1869, connecting the Mediterranean Sea to the Red Sea.", "Wikipedia"),
        ("Renaissance Era", "The Renaissance was a cultural movement that began in Florence, Italy in the 14th century and spread throughout Europe.", "Wikipedia"),
        ("American Declaration of Independence", "The United States Declaration of Independence was adopted by the Second Continental Congress on July 4, 1776.", "government"),
        ("Panama Canal Completion", "The Panama Canal was opened on August 15, 1914, connecting the Atlantic Ocean with the Pacific Ocean across the Isthmus of Panama.", "Wikipedia"),
        ("World War II End", "World War II in Europe concluded with the surrender of Nazi Germany on May 8, 1945 (V-E Day).", "Wikipedia"),
        ("Sputnik 1 Launch", "The Soviet Union launched Sputnik 1, the first artificial Earth satellite, on October 4, 1957.", "Wikipedia"),
        ("Printing Press Invention", "Johannes Gutenberg introduced movable-type printing to Europe around 1440, revolutionizing book manufacturing.", "Wikipedia"),
        ("Silk Road Trade", "The Silk Road was an ancient network of Eurasian trade routes that connected East Asia with the Mediterranean from the Han dynasty onwards.", "Wikipedia"),
        ("Discovery of Tutankhamun Tomb", "Howard Carter discovered the intact tomb of the 18th Dynasty Pharaoh Tutankhamun in November 1922 in the Valley of the Kings.", "Wikipedia"),
    ],
    "Geography": [
        ("Mount Everest Altitude", "Mount Everest is Earth's highest mountain above sea level, reaching 8,848.86 metres in the Himalayas on the border of Nepal and China.", "Wikipedia"),
        ("Amazon River Length", "The Amazon River in South America is the largest river by discharge volume of water in the world and the second longest after the Nile.", "Wikipedia"),
        ("Sahara Desert", "The Sahara is the largest hot desert in the world, covering over 9 million square kilometres across North Africa.", "Wikipedia"),
        ("Mariana Trench", "The Challenger Deep in the Mariana Trench is the deepest known point in Earth's oceans, reaching approximately 10,994 metres.", "scientific"),
        ("Lake Baikal", "Lake Baikal in southern Russia is the world's deepest and oldest freshwater lake, containing about 20% of Earth's unfrozen surface fresh water.", "scientific"),
        ("Nile River", "The Nile River flows northward through northeastern Africa and is generally considered the longest river in the world at approximately 6,650 km.", "Wikipedia"),
        ("Great Barrier Reef", "The Great Barrier Reef is the world's largest coral reef system, located in the Coral Sea off the coast of Queensland, Australia.", "scientific"),
        ("Dead Sea Elevation", "The shores of the Dead Sea in the Jordan Rift Valley are the lowest land elevation on Earth, at roughly 430 metres below sea level.", "Wikipedia"),
        ("Antarctica Ice Sheet", "The Antarctic ice sheet is the largest single mass of ice on Earth, covering an area of nearly 14 million square kilometres.", "scientific"),
        ("Greenland Landmass", "Greenland is the world's largest island that is not a continent, situated between the Arctic and Atlantic Oceans.", "Wikipedia"),
        ("Victoria Falls", "Victoria Falls is a waterfall on the Zambezi River in southern Africa between Zambia and Zimbabwe, spanning roughly 1,708 metres in width.", "Wikipedia"),
        ("Grand Canyon", "The Grand Canyon is a steep-sided canyon carved by the Colorado River in Arizona, United States, reaching up to 1,800 metres in depth.", "Wikipedia"),
        ("Caspian Sea", "The Caspian Sea is the world's largest inland body of water, categorized as either the largest lake or a full sea.", "Wikipedia"),
        ("Mount Kilimanjaro", "Mount Kilimanjaro in Tanzania is a dormant volcano and the highest single free-standing mountain peak in Africa at 5,895 metres.", "Wikipedia"),
        ("Atacama Desert Aridity", "The Atacama Desert in northern Chile is considered the driest non-polar desert in the world.", "scientific"),
        ("Bering Strait", "The Bering Strait connects the Arctic Ocean with the Bering Sea and separates the Asian and North American continents by about 82 km.", "Wikipedia"),
        ("K2 Mountain", "K2 is the second-highest mountain on Earth at 8,611 metres, located on the China-Pakistan border in the Karakoram range.", "Wikipedia"),
        ("Suez Isthmus", "The Isthmus of Suez is the narrow strip of land that connects the continents of Africa and Asia.", "Wikipedia"),
        ("Yellowstone Caldera", "The Yellowstone Caldera in Wyoming is a supervolcano volcanic caldera formed during three supereruptions over the past 2.1 million years.", "scientific"),
        ("Angel Falls", "Angel Falls in Venezuela is the world's tallest uninterrupted waterfall, with a height of 979 metres and a plunge of 807 metres.", "Wikipedia"),
    ],
    "Politics and government": [
        ("Bicameral Legislature", "A bicameral legislature is a legislative body consisting of two separate houses or chambers, such as the US Senate and House of Representatives.", "government"),
        ("Universal Declaration of Human Rights", "The Universal Declaration of Human Rights was adopted by the United Nations General Assembly in Paris on December 10, 1948.", "official_organization"),
        ("US Constitution Ratification", "The Constitution of the United States was drafted in 1787 and ratified in 1788, establishing the framework of the federal government.", "government"),
        ("European Union Parliament", "The European Parliament is the directly elected parliamentary institution of the European Union, representing EU citizens.", "official_organization"),
        ("Geneva Conventions", "The Geneva Conventions are four treaties and three protocols establishing international legal standards for humanitarian treatment during war.", "official_organization"),
        ("Constitutional Monarchy", "A constitutional monarchy is a form of monarchy in which the monarch exercises authority in accordance with a constitution.", "government"),
        ("Supreme Court of the US", "The Supreme Court of the United States consists of the Chief Justice and eight associate justices appointed for life under good behavior.", "government"),
        ("International Court of Justice", "The International Court of Justice is the principal judicial organ of the United Nations, seated in The Hague, Netherlands.", "official_organization"),
        ("Parliamentary Republic", "In a parliamentary republic, the executive branch derives its democratic legitimacy from and is accountable to the legislature.", "government"),
        ("Electoral College", "The United States Electoral College is the group of presidential electors formed every four years to elect the president and vice president.", "government"),
        ("Separation of Powers", "The doctrine of separation of powers divides governance into three independent branches: legislative, executive, and judicial.", "government"),
        ("NATO Alliance", "The North Atlantic Treaty Organization is an intergovernmental military alliance between 32 member states from Europe and North America.", "official_organization"),
        ("Cabinet Government", "A cabinet is a body of high-ranking state officials, typically consisting of top leaders of the executive branch.", "government"),
        ("Treaty of Rome", "The Treaty of Rome was signed on March 25, 1957, establishing the European Economic Community (EEC).", "official_organization"),
        ("Veto Power UN Security Council", "The five permanent members of the UN Security Council (China, France, Russia, UK, US) hold veto power over substantive resolutions.", "official_organization"),
        ("Magna Carta Rule of Law", "The Magna Carta asserted the principle of the rule of law, requiring the sovereign to be bound by legal constraints.", "government"),
        ("Federalism Concept", "Federalism is a system of government where power is divided between a central national authority and constituent political units.", "government"),
        ("Direct Democracy", "Direct democracy is a form of democracy in which citizens decide on policy initiatives directly rather than electing representatives.", "government"),
        ("Universal Suffrage", "Universal suffrage grants the right to vote to all adult citizens regardless of wealth, income, gender, or social status.", "government"),
        ("Judicial Review", "Judicial review is the power of a court to review statutory enactments and executive acts to determine constitutional validity.", "government"),
    ],
    "Economics": [
        ("Central Bank Monetary Policy", "Central banks use monetary-policy tools, including interest rates and reserve requirements, to regulate money supply and inflation.", "government"),
        ("Gross Domestic Product", "Gross Domestic Product (GDP) is a monetary measure of the market value of all final goods and services produced within a country over a specific time period.", "government"),
        ("Inflation Definition", "Inflation is the general rise in the price level of goods and services in an economy over a sustained period.", "government"),
        ("Bretton Woods Conference", "The Bretton Woods Conference in July 1944 established the International Monetary Fund and the International Bank for Reconstruction and Development.", "official_organization"),
        ("Law of Supply and Demand", "The law of supply and demand describes how the price of an asset or commodity is determined by the balance of availability and consumer desire.", "scientific"),
        ("World Bank Mission", "The World Bank is an international financial institution that provides loans and grants to low- and middle-income countries for capital projects.", "official_organization"),
        ("Fiscal Policy", "Fiscal policy refers to government spending and taxation decisions used to influence macroeconomic conditions.", "government"),
        ("Comparative Advantage", "The theory of comparative advantage, formulated by David Ricardo, shows how countries benefit from trade by specializing in goods they produce with lower opportunity costs.", "scientific"),
        ("Federal Reserve System", "The Federal Reserve is the central banking system of the United States, created on December 23, 1913 by the Federal Reserve Act.", "government"),
        ("Opportunity Cost", "Opportunity cost represents the potential benefits an individual, investor, or business misses out on when choosing one alternative over another.", "scientific"),
        ("Purchasing Power Parity", "Purchasing power parity is an economic metric comparing economic productivity and standards of living between countries.", "scientific"),
        ("Keynesian Economics", "Keynesian economics is a macroeconomic theory developed by John Maynard Keynes advocating government intervention during economic downturns.", "scientific"),
        ("Stock Market Index", "A stock market index measures a hypothetical portfolio of investment holdings to indicate performance of a market sector or economy.", "government"),
        ("Tariff Definition", "A tariff is a customs duty or tax levied on imports of goods and services by a sovereign government.", "government"),
        ("Microeconomics vs Macroeconomics", "Microeconomics analyzes individual agents and markets, while macroeconomics studies aggregated economy-wide phenomena such as unemployment and GDP.", "scientific"),
        ("Monopoly Market Structure", "A monopoly exists when a single enterprise is the only supplier of a particular commodity or service in a given market.", "scientific"),
        ("Elasticity of Demand", "Price elasticity of demand measures the responsiveness of quantity demanded to a change in price.", "scientific"),
        ("Balance of Payments", "The balance of payments records all economic transactions between residents of a country and the rest of the world over a specified period.", "government"),
        ("Cryptocurrency Decentralization", "Cryptocurrencies use decentralized consensus mechanisms on distributed ledgers to verify transactions without requiring central clearing houses.", "scientific"),
        ("Gini Coefficient", "The Gini coefficient is a statistical measure of economic inequality across a population, ranging from 0 (perfect equality) to 1 (perfect inequality).", "scientific"),
    ],
    "Sports": [
        ("Olympic Games Modern", "The first modern Olympic Games were held in Athens, Greece in April 1896, organized by the International Olympic Committee.", "official_organization"),
        ("FIFA World Cup 1930", "The inaugural FIFA World Cup took place in Uruguay in July 1930, with Uruguay winning the championship match.", "official_organization"),
        ("Marathon Distance", "The standard marathon distance is 42.195 kilometres (26 miles 385 yards), established officially at the 1908 London Olympic Games.", "official_organization"),
        ("Tour de France", "The Tour de France is an annual men's multiple-stage bicycle race primarily held in France, first organized in 1903 by Géo Lefèvre.", "Wikipedia"),
        ("Basketball Invention", "James Naismith invented the game of basketball in December 1891 in Springfield, Massachusetts, writing the original 13 rules.", "Wikipedia"),
        ("Wimbledon Championship", "The Championships, Wimbledon is the oldest tennis tournament in the world, founded in 1877 at the All England Club in London.", "official_organization"),
        ("Super Bowl History", "The first AFL-NFL World Championship Game, later known as Super Bowl I, was played on January 15, 1967 in Los Angeles.", "official_organization"),
        ("Cricket Laws", "The Laws of Cricket were first drawn up in 1744 and are maintained and copyrighted by the Marylebone Cricket Club (MCC) in London.", "official_organization"),
        ("Summer Olympics Frequency", "The Summer Olympic Games are held every four years, with the exception of cancellations during World Wars I and II and the 2020 postponement.", "official_organization"),
        ("World Athletics Record 100m", "Usain Bolt of Jamaica set the world record in the 100 metres sprint at 9.58 seconds in Berlin on August 16, 2009.", "official_organization"),
        ("Formula One Championship", "The Formula One World Championship was founded in 1950, with the first race held at the Silverstone Circuit in England.", "official_organization"),
        ("Rugby World Cup", "The inaugural Rugby World Cup took place in 1987, co-hosted by New Zealand and Australia, with New Zealand winning the title.", "official_organization"),
        ("Ryder Cup", "The Ryder Cup is a biennial men's golf competition between teams from Europe and the United States, initiated in 1927.", "official_organization"),
        ("Paralympic Games", "The first official Paralympic Games were held in Rome in 1960, bringing together 400 athletes from 23 countries.", "official_organization"),
        ("Volleyball Invention", "William G. Morgan invented volleyball in Holyoke, Massachusetts in 1895, originally naming the sport Mintonette.", "Wikipedia"),
        ("Decathlon Events", "The Olympic decathlon consists of ten track and field events held over two consecutive days.", "official_organization"),
        ("Baseball World Series", "The World Series is the annual championship series of Major League Baseball, first played in 1903 between Boston and Pittsburgh.", "official_organization"),
        ("Stanley Cup", "The Stanley Cup is the oldest existing trophy awarded to a professional sports franchise in North America, commissioned in 1892.", "official_organization"),
        ("Badminton Scoring", "In international badminton, a match consists of the best of three games of 21 points using rally scoring.", "official_organization"),
        ("PGA Tour", "The PGA Tour is the organizer of the main professional golf tours played by men in the United States and North America.", "official_organization"),
    ],
    "Medicine and health": [
        ("Penicillin Discovery Fleming", "Penicillin was discovered in September 1928 by Alexander Fleming at St Mary's Hospital in London, revolutionizing antibacterial treatment.", "scientific"),
        ("Smallpox Eradication", "Smallpox was officially declared eradicated by the World Health Organization in May 1980 following a global vaccination campaign.", "official_organization"),
        ("Insulin Discovery", "Insulin was discovered in 1921 by Frederick Banting and Charles Best working in the laboratory of J. J. R. Macleod at the University of Toronto.", "scientific"),
        ("Polio Vaccine Salk", "Jonas Salk developed the first successful inactivated polio vaccine, announced as safe and effective on April 12, 1955.", "scientific"),
        ("World Health Organization Function", "The World Health Organization is a specialized agency of the United Nations responsible for international public health.", "official_organization"),
        ("Human Genome Project", "The Human Genome Project was an international research project completed in April 2003 that sequenced the vast majority of human DNA base pairs.", "scientific"),
        ("Antibiotic Resistance", "Antibiotic resistance occurs when bacteria adapt and survive exposure to antibiotics designed to inhibit or kill them.", "scientific"),
        ("Cardiovascular System", "The cardiovascular system consists of the heart, blood vessels, and blood, functioning to transport oxygen and nutrients to tissues throughout the body.", "scientific"),
        ("Tuberculosis Causative Agent", "Tuberculosis is an infectious disease caused by Mycobacterium tuberculosis, identified by Robert Koch in 1882.", "scientific"),
        ("Hemoglobin Function", "Hemoglobin is an iron-containing protein in red blood cells that transports oxygen from the respiratory organs to the rest of the body.", "scientific"),
        ("Immune System T Cells", "T lymphocytes are a type of white blood cell of the immune system that mature in the thymus gland and participate in cell-mediated immunity.", "scientific"),
        ("Aspirin History", "Felix Hoffmann synthesized acetylsalicylic acid (aspirin) in 1897 at the Bayer company in Germany.", "scientific"),
        ("Diabetes Types", "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin, whereas Type 2 involves insulin resistance.", "scientific"),
        ("Blood Pressure Measurement", "Blood pressure is recorded as systolic pressure over diastolic pressure, measured in millimetres of mercury (mmHg).", "scientific"),
        ("DNA Structure Discovery", "James Watson, Francis Crick, and Maurice Wilkins were awarded the 1962 Nobel Prize in Physiology or Medicine for nucleic acid structural discoveries.", "scientific"),
        ("Vaccination Principle", "Vaccination trains the immune system to recognize and fight pathogens by exposing it to an antigen derived from the microbe.", "scientific"),
        ("X-Ray Discovery", "Wilhelm Conrad Rontgen discovered X-rays in 1895, receiving the first Nobel Prize in Physics in 1901 for this breakthrough.", "scientific"),
        ("Malaria Parasite", "Malaria is a mosquito-borne infectious disease caused by parasitic protozoans of the genus Plasmodium.", "scientific"),
        ("Clinical Trial Phases", "Clinical drug development proceeds through Phase I (safety), Phase II (efficacy/dosing), Phase III (large-scale confirmation), and Phase IV (post-marketing).", "official_organization"),
        ("Hippocratic Oath", "The Hippocratic Oath is an oath of ethics historically taken by physicians, traditionally attributed to Hippocrates of Kos.", "Wikipedia"),
    ],
    "Environment": [
        ("Paris Climate Agreement", "The Paris Agreement is a legally binding international treaty on climate change adopted by 196 parties at COP 21 in Paris on December 12, 2015.", "official_organization"),
        ("Greenhouse Effect", "The greenhouse effect is the process by which radiation from a planet's atmosphere warms the planet's surface to a temperature above what it would be without its atmosphere.", "scientific"),
        ("Carbon Cycle", "The carbon cycle is the biogeochemical cycle by which carbon is exchanged among the biosphere, pedosphere, geosphere, hydrosphere, and atmosphere of Earth.", "scientific"),
        ("Deforestation Impact", "Deforestation contributes to habitat loss, decreased biodiversity, and the release of sequestered carbon into the atmosphere.", "scientific"),
        ("Renewable Energy Sources", "Renewable energy comes from natural sources that are constantly replenished, including solar, wind, hydro, geothermal, and biomass energy.", "scientific"),
        ("Biodiversity Hotspots", "A biodiversity hotspot is a biogeographic region with significant levels of biodiversity that is threatened by human habitation.", "scientific"),
        ("Ocean Acidification", "Ocean acidification is the ongoing decrease in the pH of Earth's oceans caused by the uptake of anthropogenic carbon dioxide from the atmosphere.", "scientific"),
        ("Kyoto Protocol", "The Kyoto Protocol was an international treaty adopted on December 11, 1997, committing industrialized countries to reduce greenhouse gas emissions.", "official_organization"),
        ("Intergovernmental Panel on Climate Change", "The IPCC was created in 1988 by the World Meteorological Organization and the United Nations Environment Programme to provide scientific assessments on climate change.", "official_organization"),
        ("Wetlands Ecological Role", "Wetlands act as natural water filters, mitigate flood risks, and provide critical habitats for migratory birds and aquatic species.", "scientific"),
        ("Plastic Pollution in Oceans", "Plastic waste in marine environments breaks down into microplastics, which enter the marine food web and cause broad ecological damage.", "scientific"),
        ("Endangered Species Act", "The United States Endangered Species Act of 1973 provides a framework to conserve and protect endangered and threatened species and their habitats.", "government"),
        ("Tundra Ecosystem", "Tundra is a biome where tree growth is hindered by low temperatures and short growing seasons, characterized by permafrost soils.", "scientific"),
        ("Amazon Rainforest Carbon Sink", "The Amazon basin contains the largest tropical rainforest in the world and stores an estimated 150-200 billion tonnes of carbon.", "scientific"),
        ("Montréal Protocol", "The Montreal Protocol on Substances that Deplete the Ozone Layer was agreed in 1987, successfully phasing out the production of chlorofluorocarbons (CFCs).", "official_organization"),
        ("Eutrophication Process", "Eutrophication occurs when a water body becomes overly enriched with nutrients, leading to excessive algae growth and hypoxia.", "scientific"),
        ("Coral Bleaching Causes", "Coral bleaching occurs when corals expel symbiotic zooxanthellae algae due to elevated water temperatures or pollution.", "scientific"),
        ("Ramsar Convention", "The Ramsar Convention is an international treaty for the conservation and sustainable use of wetlands, signed in Ramsar, Iran in 1971.", "official_organization"),
        ("Soil Degradation", "Soil erosion and nutrient depletion reduce agricultural productivity and undermine terrestrial ecosystem stability worldwide.", "scientific"),
        ("Species Extinction Rates", "Current species extinction rates are estimated by conservation biologists to be 100 to 1,000 times higher than the natural background rate.", "scientific"),
    ],
    "Education": [
        ("Formative Assessment", "Formative assessment takes place during instruction and is intended to provide feedback that can guide teaching and improve learning.", "other"),
        ("Summative Assessment", "Summative assessment evaluates student learning at the end of an instructional unit by comparing it against some standard or benchmark.", "other"),
        ("Bloom's Taxonomy", "Bloom's Taxonomy, published in 1956 by Benjamin Bloom and collaborators, classifies educational learning objectives into cognitive complexity levels.", "scientific"),
        ("Montessori Education", "Montessori education is a child-centered educational approach developed by Italian physician Maria Montessori in the early 20th century.", "Wikipedia"),
        ("Pedagogy Definition", "Pedagogy is the method and practice of teaching, especially as an academic subject or theoretical concept.", "other"),
        ("Higher Education Bologna Process", "The Bologna Process is a series of agreements between European countries to ensure comparability in the standards and quality of higher-education qualifications.", "official_organization"),
        ("Vygotsky Scaffolding", "Lev Vygotsky introduced the concept of the Zone of Proximal Development, describing what a learner can do with guidance versus independently.", "scientific"),
        ("Peer Assessment", "Peer assessment involves students evaluating the work of their peers based on predefined criteria and performance standards.", "other"),
        ("University of Bologna", "The University of Bologna in Italy, founded in 1088, is widely considered the oldest continuously operating university in the world.", "Wikipedia"),
        ("Inclusive Education", "Inclusive education ensures that students with diverse learning needs and disabilities are educated in general classroom environments alongside their peers.", "official_organization"),
        ("STEM Education", "STEM education integrates science, technology, engineering, and mathematics into an interdisciplinary and applied curriculum.", "government"),
        ("PISA International Assessment", "The Programme for International Student Assessment (PISA) is a worldwide study by the OECD measuring 15-year-old school pupils' scholastic performance.", "official_organization"),
        ("Jean Piaget Stages of Development", "Jean Piaget proposed four stages of cognitive development: sensorimotor, preoperational, concrete operational, and formal operational.", "scientific"),
        ("Distance Learning Evolution", "Distance education has evolved from correspondence courses using postal mail to modern synchronous and asynchronous digital e-learning platforms.", "other"),
        ("Curriculum Development", "Curriculum design involves organizing learning goals, instructional activities, and evaluation methods into a coherent instructional sequence.", "other"),
        ("Adult Learning Theory Andragogy", "Malcolm Knowles popularized andragogy, arguing that adults learn differently from children due to self-directedness and accumulated life experience.", "scientific"),
        ("Standardized Testing", "Standardized tests require all test takers to answer the same questions in the same way, scored in a consistent manner to compare relative performance.", "other"),
        ("Oxford University", "The University of Oxford in England has evidence of teaching as early as 1096, making it the oldest university in the English-speaking world.", "Wikipedia"),
        ("Gamification in Learning", "Educational gamification incorporates game design elements into non-game learning environments to enhance engagement and motivation.", "scientific"),
        ("Open Educational Resources", "Open Educational Resources are teaching and research materials in the public domain or released under an open license permitting free reuse.", "official_organization"),
    ],
    "Space": [
        ("Mars Satellites Phobos Deimos", "Mars has two small natural satellites, Phobos and Deimos, discovered by Asaph Hall in August 1877.", "scientific"),
        ("Jupiter Great Red Spot", "Jupiter's Great Red Spot is a persistent high-pressure storm in the planet's atmosphere, observed continuously since at least 1878.", "scientific"),
        ("Saturn Ring System", "Saturn's planetary ring system is the most extensive in the Solar System, composed predominantly of water ice particles with trace rock debris.", "scientific"),
        ("James Webb Space Telescope", "The James Webb Space Telescope was launched on December 25, 2021, operating near the Sun-Earth L2 Lagrange point to observe infrared astronomy.", "scientific"),
        ("International Space Station", "The International Space Station is a modular space station in low Earth orbit, continuously occupied by international crews since November 2000.", "official_organization"),
        ("Voyager 1 Interstellar Mission", "NASA's Voyager 1 probe, launched in September 1977, entered interstellar space in August 2012, becoming the most distant human-made object.", "scientific"),
        ("Pluto Reclassification", "The International Astronomical Union reclassified Pluto from a planet to a dwarf planet in August 2006.", "official_organization"),
        ("Solar System Planets Count", "The Solar System contains eight recognized planets orbiting the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.", "scientific"),
        ("Hubble Deep Field", "The Hubble Deep Field image, captured in 1995, revealed thousands of galaxies in a tiny region of space previously thought to be mostly empty.", "scientific"),
        ("Venus Atmosphere", "Venus has an extremely dense atmosphere composed mainly of carbon dioxide with clouds of sulfuric acid, creating a runaway greenhouse effect.", "scientific"),
        ("Exoplanet Detection Kepler", "NASA's Kepler space telescope discovered thousands of confirmed exoplanets using the transit photometry method between 2009 and 2018.", "scientific"),
        ("Supermassive Black Hole Sagittarius A*", "Sagittarius A* is a supermassive black hole at the Galactic Center of the Milky Way, with a mass of roughly 4 million solar masses.", "scientific"),
        ("Lunar Phases Mechanism", "Lunar phases result from the changing geometry between the Sun, Earth, and Moon as the Moon orbits Earth roughly every 29.5 days.", "scientific"),
        ("Kuiper Belt", "The Kuiper Belt is a circumstellar disc in the outer Solar System extending beyond Neptune's orbit, containing icy bodies and dwarf planets.", "scientific"),
        ("Apollo 17 Mission", "Apollo 17 was the final mission of NASA's Apollo program, landing Eugene Cernan and Harrison Schmitt on the Moon in December 1972.", "scientific"),
        ("Comet Halley Orbit", "Halley's Comet is a periodic comet with an orbital period of approximately 75 to 76 years, last seen in the inner Solar System in 1986.", "scientific"),
        ("Mars Curiosity Rover", "NASA's Curiosity rover landed in Gale Crater on Mars in August 2012 to investigate past Martian habitability and geology.", "scientific"),
        ("Solar Flares", "Solar flares are intense localized eruptions of electromagnetic radiation in the Sun's atmosphere caused by magnetic field reconnection.", "scientific"),
        ("Oort Cloud", "The Oort cloud is a theoretical cloud of predominantly icy planetesimals proposed to surround the Sun at distances up to 100,000 AU.", "scientific"),
        ("Titan Moon of Saturn", "Titan is Saturn's largest moon and the only known moon with a dense atmosphere and stable surface liquid hydrocarbon lakes.", "scientific"),
    ],
    "Companies and organizations": [
        ("World Health Organization Structure", "The World Health Organization is a specialized agency of the United Nations responsible for international public health, founded in 1948.", "official_organization"),
        ("Apple Inc Founding", "Apple Inc. was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne to develop and sell the Apple I personal computer.", "Wikipedia"),
        ("Microsoft Founding", "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen to develop and sell BASIC interpreters for the Altair 8800.", "Wikipedia"),
        ("Google Founding", "Google was founded on September 4, 1998 by Larry Page and Sergey Brin while they were PhD students at Stanford University.", "Wikipedia"),
        ("Amazon Founding", "Amazon was founded on July 5, 1994 by Jeff Bezos in Bellevue, Washington, starting as an online marketplace for books.", "Wikipedia"),
        ("International Monetary Fund", "The IMF is an international financial organization of 190 member countries working to foster global monetary cooperation and financial stability.", "official_organization"),
        ("Red Cross Movement", "The International Red Cross and Red Crescent Movement was founded in 1863 in Geneva, Switzerland by Henry Dunant.", "official_organization"),
        ("CERN Laboratory", "CERN is the European Organization for Nuclear Research, operating the largest particle physics laboratory in the world near Geneva.", "official_organization"),
        ("UNESCO Organization", "UNESCO is a specialized agency of the United Nations aimed at promoting world peace and security through international cooperation in education and science.", "official_organization"),
        ("Toyota Motor Corporation", "Toyota Motor Corporation was founded on August 28, 1937 by Kiichiro Toyoda as a spinoff from his father's company Toyoda Automatic Loom Works.", "Wikipedia"),
        ("IBM History", "IBM was founded on June 16, 1911 as the Computing-Tabulating-Recording Company (CTR) and renamed International Business Machines in 1924.", "Wikipedia"),
        ("World Trade Organization", "The World Trade Organization was established on January 1, 1995 under the 1994 Marrakesh Agreement, replacing the GATT.", "official_organization"),
        ("SpaceX Founding", "SpaceX was founded in March 2002 by Elon Musk with the goal of reducing space transportation costs to enable the colonization of Mars.", "Wikipedia"),
        ("Amnesty International", "Amnesty International was founded in London in July 1961 by lawyer Peter Benenson to campaign for human rights globally.", "official_organization"),
        ("Sony Corporation", "Sony was established on May 7, 1946 by Masaru Ibuka and Akio Morita in Tokyo, Japan, originally under the name Tokyo Tsushin Kogyo.", "Wikipedia"),
        ("Doctors Without Borders", "Médecins Sans Frontières (Doctors Without Borders) was founded in 1971 by French doctors and journalists following the Nigerian Civil War.", "official_organization"),
        ("OpenAI Founding", "OpenAI was founded in December 2015 as a non-profit artificial intelligence research organization in San Francisco, California.", "Wikipedia"),
        ("Boeing Company", "The Boeing Company was founded by William Boeing in Seattle, Washington on July 15, 1916, becoming a major aerospace manufacturer.", "Wikipedia"),
        ("World Wildlife Fund", "WWF was founded on April 29, 1961 in Morges, Switzerland, working in the field of wilderness preservation and endangered species.", "official_organization"),
        ("Intel Corporation", "Intel was founded on July 18, 1968 by semiconductor pioneers Robert Noyce and Gordon Moore in Mountain View, California.", "Wikipedia"),
    ],
    "Famous people": [
        ("William Shakespeare Hamlet", "William Shakespeare was an English playwright and poet who wrote tragedies including Hamlet, Othello, King Lear, and Macbeth.", "Wikipedia"),
        ("Albert Einstein Nobel Prize", "Albert Einstein was awarded the 1921 Nobel Prize in Physics for his services to Theoretical Physics, and especially for his discovery of the law of the photoelectric effect.", "scientific"),
        ("Marie Curie Nobel Prizes", "Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields: Physics (1903) and Chemistry (1911).", "scientific"),
        ("Isaac Newton Principia", "Sir Isaac Newton published Philosophiae Naturalis Principia Mathematica in 1687, formulating the laws of motion and universal gravitation.", "scientific"),
        ("Leonardo da Vinci Mona Lisa", "Leonardo da Vinci was an Italian polymath of the Renaissance who painted masterpieces including the Mona Lisa and The Last Supper.", "Wikipedia"),
        ("Nelson Mandela Presidency", "Nelson Mandela was a South African anti-apartheid revolutionary who served as President of South Africa from 1994 to 1999.", "Wikipedia"),
        ("Charles Darwin Origin of Species", "Charles Darwin published On the Origin of Species in 1859, introducing the scientific theory of evolution by natural selection.", "scientific"),
        ("Ada Lovelace Computer Program", "Ada Lovelace wrote the first algorithm intended to be carried out by Charles Babbage's mechanical Analytical Engine in 1843.", "Wikipedia"),
        ("Mahatma Gandhi Nonviolence", "Mahatma Gandhi led the campaign for Indian independence through nonviolent civil disobedience, inspiring movements for civil rights worldwide.", "Wikipedia"),
        ("Galileo Galilei Telescopic Observations", "Galileo Galilei made groundbreaking astronomical observations with his telescope in 1609-1610, discovering four largest moons of Jupiter.", "scientific"),
        ("Alan Turing Computing Legacy", "Alan Turing played a pivotal role in cracking intercepted coded messages at Bletchley Park and is considered the father of theoretical computer science.", "scientific"),
        ("Frida Kahlo Art", "Frida Kahlo was a Mexican painter known for her many self-portraits and works inspired by the nature and artifacts of Mexico.", "Wikipedia"),
        ("Ludwig van Beethoven Symphonies", "Ludwig van Beethoven was a German composer who composed nine symphonies, creating landmark works during the transition between the Classical and Romantic eras.", "Wikipedia"),
        ("Nikola Tesla Alternating Current", "Nikola Tesla contributed significantly to the design of the modern alternating current (AC) electricity supply system.", "scientific"),
        ("Jane Goodall Primatology", "Jane Goodall is an English primatologist and anthropologist renowned for her landmark 60-year study of chimpanzee social interactions in Gombe Stream National Park.", "scientific"),
        ("Winston Churchill Leadership", "Winston Churchill served as Prime Minister of the United Kingdom from 1940 to 1945 during the Second World War.", "Wikipedia"),
        ("Ernest Hemingway Literature", "Ernest Hemingway was an American novelist and short-story writer who received the Nobel Prize in Literature in 1954.", "Wikipedia"),
        ("Aristotle Philosophy", "Aristotle was an Ancient Greek philosopher whose writings covered physics, biology, zoology, metaphysics, ethics, and poetry.", "Wikipedia"),
        ("Martin Luther King Jr Speech", "Martin Luther King Jr. delivered his 'I Have a Dream' speech on August 28, 1963 during the March on Washington for Jobs and Freedom.", "Wikipedia"),
        ("Steve Jobs Apple Leadership", "Steve Jobs co-founded Apple Inc. and later oversaw the creation of revolutionary products including the iMac, iPod, iPhone, and iPad.", "Wikipedia"),
    ],
    "Culture": [
        ("Haiku Poetic Form", "Haiku is a traditional Japanese poetic form consisting of three phrases with a 5, 7, 5 syllable structure, often focusing on nature.", "Wikipedia"),
        ("Renaissance Art Characteristics", "Renaissance art originated in Italy and is characterized by naturalism, mathematical perspective, and classical themes.", "Wikipedia"),
        ("Gothic Architecture", "Gothic architecture originated in 12th-century France, distinguished by pointed arches, ribbed vaults, and flying buttresses.", "Wikipedia"),
        ("Symphony Orchestra Structure", "A symphony orchestra is a large ensemble of musicians consisting of four instrumental families: strings, woodwinds, brass, and percussion.", "Wikipedia"),
        ("Origami Paper Folding", "Origami is the traditional Japanese art of paper folding, creating intricate representations without using scissors or glue.", "Wikipedia"),
        ("Kabuki Theatre", "Kabuki is a classical Japanese dance-drama known for its stylized performance, elaborate costumes, and distinctive kumadori makeup.", "Wikipedia"),
        ("Tango Dance Origin", "Tango is a partner dance and musical genre that originated in the late 19th century in the border areas between Argentina and Uruguay.", "Wikipedia"),
        ("Sonnet Poetic Structure", "A traditional Petrarchan or Shakespearean sonnet consists of 14 lines written in iambic pentameter with a specific rhyming scheme.", "Wikipedia"),
        ("Impressionist Painting", "Impressionism was a 19th-century art movement in France characterized by visible brush strokes, open composition, and emphasis on accurate depiction of light.", "Wikipedia"),
        ("Flamenco Music and Dance", "Flamenco is an art form based on the varied folkloric music traditions of southern Spain in the autonomous community of Andalusia.", "Wikipedia"),
        ("Jazz Music Origins", "Jazz originated in the African-American communities of New Orleans in the late 19th and early 20th centuries, rooted in blues and ragtime.", "Wikipedia"),
        ("Baroque Architecture", "Baroque architecture is a highly decorative style that originated in late 16th-century Italy, characterized by dynamic forms and dramatic lighting.", "Wikipedia"),
        ("UNESCO World Heritage Sites", "UNESCO designates World Heritage Sites of outstanding cultural or natural importance to the common heritage of humanity.", "official_organization"),
        ("Calligraphy Art", "Calligraphy is a visual art related to writing, designing and executing lettering with broad-tipped instruments or brushes.", "Wikipedia"),
        ("Opera Art Form", "Opera is an art form in which singers and musicians perform a dramatic work combining text (libretto) and musical score in a theatrical setting.", "Wikipedia"),
        ("Culinary Heritage UNESCO", "UNESCO recognizes traditional gastronomic cultures, including the Mediterranean diet and French gastronomic meal, as Intangible Cultural Heritage.", "official_organization"),
        ("Carnival Celebrations", "Carnival is a Western Christian festive season occurring before Lent, celebrated with parades and masquerade balls, notably in Rio de Janeiro and Venice.", "Wikipedia"),
        ("Surrealism Movement", "Surrealism was a 20th-century avant-garde movement in art and literature that sought to release the creative potential of the unconscious mind.", "Wikipedia"),
        ("Cinema Lumiere Brothers", "Auguste and Louis Lumiere patented the Cinematographe in 1895 and presented the first commercial public screening of projected motion pictures in Paris.", "Wikipedia"),
        ("Ballet Classical Dance", "Classical ballet is a formalized type of dance originating during the Italian Renaissance in the 15th century and developing into concert dance in France and Russia.", "Wikipedia"),
    ],
    "General knowledge": [
        ("Water Freezing and Boiling Points", "Under standard atmospheric pressure (1 atm), pure water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.", "scientific"),
        ("Earth Rotation Period", "Earth rotates once on its axis relative to distant stars in approximately 23 hours, 56 minutes, and 4 seconds (one sidereal day).", "scientific"),
        ("Leap Year Rule", "In the Gregorian calendar, a leap year occurs every four years, except for years divisible by 100 that are not also divisible by 400.", "Wikipedia"),
        ("Alphabet Letters Count English", "The modern English alphabet contains 26 letters, consisting of five vowels (A, E, I, O, U) and 21 consonants.", "Wikipedia"),
        ("Continents Number", "In the conventional seven-continent model, Earth's continents are Asia, Africa, North America, South America, Antarctica, Europe, and Australia.", "Wikipedia"),
        ("Prime Number Definition", "A prime number is a natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers.", "scientific"),
        ("Solar Eclipse Alignment", "A solar eclipse occurs when the Moon passes between Earth and the Sun, thereby fully or partially obscuring the view of the Sun from Earth.", "scientific"),
        ("Atmospheric Composition", "Earth's atmosphere by dry volume consists of approximately 78.08% nitrogen, 20.95% oxygen, 0.93% argon, and trace other gases.", "scientific"),
        ("International Date Line", "The International Date Line is an internationally recognized calendar line on Earth's surface roughly following the 180th meridian of longitude.", "Wikipedia"),
        ("Metric System SI Units", "The International System of Units (SI) defines seven fundamental base units: second, metre, kilogram, ampere, kelvin, mole, and candela.", "scientific"),
        ("Earth Moon Distance", "The average distance from the center of Earth to the center of the Moon is approximately 384,400 kilometres.", "scientific"),
        ("Human Chromosome Count", "Typical human cells contain 46 chromosomes organized into 23 homologous pairs, including two sex chromosomes.", "scientific"),
        ("Diamond Hardness Mohs", "Diamond is the hardest naturally occurring substance known, rating 10 on the Mohs scale of mineral hardness.", "scientific"),
        ("Sound Speed in Air", "The speed of sound in dry air at 20 degrees Celsius is approximately 343 metres per second (1,235 km/h).", "scientific"),
        ("Primary Colors of Light", "In additive color mixing (such as on screens), the primary colors are red, green, and blue (RGB).", "scientific"),
        ("Richter Magnitude Scale", "The Richter scale is a logarithmic scale developed in 1935 by Charles F. Richter to quantify the energy released by earthquakes.", "scientific"),
        ("Rainbow Optical Phenomenon", "A rainbow is a meteorological optical phenomenon caused by reflection, refraction, and dispersion of light in water droplets.", "scientific"),
        ("Gold Chemical Element", "Gold is a chemical element with symbol Au and atomic number 79, known for its corrosion resistance and high electrical conductivity.", "scientific"),
        ("Compass Magnetic North", "A magnetic compass points towards the Earth's magnetic north pole, which differs slightly from the geographic North Pole.", "scientific"),
        ("Photosynthesis Oxygen Generation", "During photosynthesis, plants utilize water and carbon dioxide in the presence of sunlight to produce glucose and release oxygen as a byproduct.", "scientific"),
    ]
}

# ── 1. Build Passage Pool ───────────────────────────────────────────────────
passage_pool = []
pid_counter = 1
passage_by_topic = {topic: [] for topic in TOPIC_PASSAGES}

for topic, p_list in TOPIC_PASSAGES.items():
    for title, text, src in p_list:
        pid = f"P{pid_counter:04d}"
        entry = {
            "passage_id": pid,
            "title": title,
            "passage_text": text,
            "source_category": src,
            "topic": topic,
            "human_verified": False,
            "source_note": "synthetic_llm_authored",
        }
        passage_pool.append(entry)
        passage_by_topic[topic].append(entry)
        pid_counter += 1

print(f"Passage pool initialized with {len(passage_pool)} passages across {len(TOPIC_PASSAGES)} topics.")

# ── 2. Read Layer 1 NLI train claims and match candidates ───────────────────
l1_train_path = os.path.join(BASE, "layer1_nli", "train.csv")
l1_claims = []
if os.path.exists(l1_train_path):
    with open(l1_train_path, encoding="utf-8", newline="") as f:
        l1_claims = list(csv.DictReader(f))

print(f"Read {len(l1_claims)} claims from layer1_nli/train.csv")

# Create candidates for each Layer 1 claim
claim_candidates_rows = []
all_pids = [p["passage_id"] for p in passage_pool]

for c in l1_claims:
    cid = c.get("id", "")
    claim_text = c.get("claim", "")
    evidence_text = c.get("evidence", "")
    label = c.get("label", "")
    topic = c.get("topic", "General knowledge")
    if topic not in passage_by_topic:
        topic = "General knowledge"
    
    # 1. Gold candidate
    gold_role = "GOLD_SUPPORT" if label == "SUPPORTS" else ("GOLD_REFUTE" if label == "REFUTES" else "PARTIAL")
    gold_pid = f"P_GOLD_{cid}"
    claim_candidates_rows.append({
        "claim_id": cid,
        "claim_text": claim_text,
        "candidate_passage_id": gold_pid,
        "candidate_passage_text": evidence_text,
        "relevance_label": "1",
        "evidence_role": gold_role,
    })
    
    # 2. Partial / Related passage in same topic
    topic_pool = passage_by_topic.get(topic, passage_pool)
    partial_p = random.choice(topic_pool)
    claim_candidates_rows.append({
        "claim_id": cid,
        "claim_text": claim_text,
        "candidate_passage_id": partial_p["passage_id"],
        "candidate_passage_text": partial_p["passage_text"],
        "relevance_label": "0",
        "evidence_role": "PARTIAL",
    })
    
    # 3. Topically related passage
    topical_p = random.choice(topic_pool)
    claim_candidates_rows.append({
        "claim_id": cid,
        "claim_text": claim_text,
        "candidate_passage_id": topical_p["passage_id"],
        "candidate_passage_text": topical_p["passage_text"],
        "relevance_label": "0",
        "evidence_role": "TOPICALLY_RELATED",
    })
    
    # 4 & 5. Irrelevant passages from other topics
    other_topics = [t for t in TOPIC_PASSAGES if t != topic]
    for _ in range(2):
        rand_t = random.choice(other_topics)
        rand_p = random.choice(passage_by_topic[rand_t])
        claim_candidates_rows.append({
            "claim_id": cid,
            "claim_text": claim_text,
            "candidate_passage_id": rand_p["passage_id"],
            "candidate_passage_text": rand_p["passage_text"],
            "relevance_label": "0",
            "evidence_role": "IRRELEVANT",
        })

# ── 3. Generate Retrieval Test Set (50 claims x 10 candidates) ──────────────
TEST_QUERIES = [
    ("RET_TEST_001", "The point-contact transistor was invented at Bell Labs in 1947.", "Technology", "The point-contact transistor was invented in December 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs."),
    ("RET_TEST_002", "Linus Torvalds initially released the Linux operating system kernel in 1991.", "Technology", "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, first released by Linus Torvalds in September 1991."),
    ("RET_TEST_003", "James Watson and Francis Crick determined the double helix structure of DNA.", "Science", "The double helix structure of DNA was determined in 1953 by James Watson and Francis Crick, using X-ray diffraction data from Rosalind Franklin."),
    ("RET_TEST_004", "Dmitri Mendeleev published the first widely recognized periodic table in 1869.", "Science", "Dmitri Mendeleev formulated the periodic law and published the first widely recognized periodic table in 1869."),
    ("RET_TEST_005", "The Berlin Wall fell on November 9, 1989.", "History", "The Berlin Wall fell on November 9, 1989, symbolizing the collapse of Soviet-backed communist regimes in Eastern Europe."),
    ("RET_TEST_006", "The Magna Carta was granted in 1215 by King John of England.", "History", "The Magna Carta was signed in June 1215 at Runnymede by King John of England, establishing that monarchial rule was subject to law."),
    ("RET_TEST_007", "Lake Baikal in Russia is the deepest freshwater lake on Earth.", "Geography", "Lake Baikal in southern Russia is the world's deepest and oldest freshwater lake, containing about 20% of Earth's unfrozen surface fresh water."),
    ("RET_TEST_008", "The Challenger Deep in the Mariana Trench is the deepest known point in the ocean.", "Geography", "The Challenger Deep in the Mariana Trench is the deepest known point in Earth's oceans, reaching approximately 10,994 metres."),
    ("RET_TEST_009", "The Universal Declaration of Human Rights was adopted by the UN in 1948.", "Politics and government", "The Universal Declaration of Human Rights was adopted by the United Nations General Assembly in Paris on December 10, 1948."),
    ("RET_TEST_010", "A bicameral legislature consists of two distinct chambers.", "Politics and government", "A bicameral legislature is a legislative body consisting of two separate houses or chambers, such as the US Senate and House of Representatives."),
    ("RET_TEST_011", "Gross Domestic Product measures the total monetary value of finished goods and services.", "Economics", "Gross Domestic Product (GDP) is a monetary measure of the market value of all final goods and services produced within a country over a specific time period."),
    ("RET_TEST_012", "David Ricardo formulated the economic theory of comparative advantage.", "Economics", "The theory of comparative advantage, formulated by David Ricardo, shows how countries benefit from trade by specializing in goods they produce with lower opportunity costs."),
    ("RET_TEST_013", "The inaugural modern Olympic Games took place in Athens in 1896.", "Sports", "The first modern Olympic Games were held in Athens, Greece in April 1896, organized by the International Olympic Committee."),
    ("RET_TEST_014", "The official marathon distance was established at 42.195 kilometres.", "Sports", "The standard marathon distance is 42.195 kilometres (26 miles 385 yards), established officially at the 1908 London Olympic Games."),
    ("RET_TEST_015", "Frederick Banting and Charles Best discovered insulin at the University of Toronto in 1921.", "Medicine and health", "Insulin was discovered in 1921 by Frederick Banting and Charles Best working in the laboratory of J. J. R. Macleod at the University of Toronto."),
    ("RET_TEST_016", "Jonas Salk developed the first safe and effective inactivated polio vaccine in 1955.", "Medicine and health", "Jonas Salk developed the first successful inactivated polio vaccine, announced as safe and effective on April 12, 1955."),
    ("RET_TEST_017", "The Paris Agreement on climate change was adopted at COP 21 in December 2015.", "Environment", "The Paris Agreement is a legally binding international treaty on climate change adopted by 196 parties at COP 21 in Paris on December 12, 2015."),
    ("RET_TEST_018", "The Montreal Protocol was designed to phase out substances that deplete the ozone layer.", "Environment", "The Montreal Protocol on Substances that Deplete the Ozone Layer was agreed in 1987, successfully phasing out the production of chlorofluorocarbons (CFCs)."),
    ("RET_TEST_019", "Bloom's Taxonomy classifies educational learning objectives into cognitive levels.", "Education", "Bloom's Taxonomy, published in 1956 by Benjamin Bloom and collaborators, classifies educational learning objectives into cognitive complexity levels."),
    ("RET_TEST_020", "Maria Montessori developed the child-centered educational approach in the early 20th century.", "Education", "Montessori education is a child-centered educational approach developed by Italian physician Maria Montessori in the early 20th century."),
    ("RET_TEST_021", "The James Webb Space Telescope operates near the Sun-Earth L2 Lagrange point.", "Space", "The James Webb Space Telescope was launched on December 25, 2021, operating near the Sun-Earth L2 Lagrange point to observe infrared astronomy."),
    ("RET_TEST_022", "NASA's Voyager 1 probe crossed into interstellar space in August 2012.", "Space", "NASA's Voyager 1 probe, launched in September 1977, entered interstellar space in August 2012, becoming the most distant human-made object."),
    ("RET_TEST_023", "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen.", "Companies and organizations", "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen to develop and sell BASIC interpreters for the Altair 8800."),
    ("RET_TEST_024", "Henry Dunant founded the International Red Cross Movement in Geneva in 1863.", "Companies and organizations", "The International Red Cross and Red Crescent Movement was founded in 1863 in Geneva, Switzerland by Henry Dunant."),
    ("RET_TEST_025", "Marie Curie was awarded Nobel Prizes in both Physics and Chemistry.", "Famous people", "Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields: Physics (1903) and Chemistry (1911)."),
    ("RET_TEST_026", "Sir Isaac Newton formulated the laws of motion in the Principia Mathematica.", "Famous people", "Sir Isaac Newton published Philosophiae Naturalis Principia Mathematica in 1687, formulating the laws of motion and universal gravitation."),
    ("RET_TEST_027", "A traditional Japanese haiku poem follows a 5-7-5 syllable structure.", "Culture", "Haiku is a traditional Japanese poetic form consisting of three phrases with a 5, 7, 5 syllable structure, often focusing on nature."),
    ("RET_TEST_028", "Tango dance originated along the border areas between Argentina and Uruguay.", "Culture", "Tango is a partner dance and musical genre that originated in the late 19th century in the border areas between Argentina and Uruguay."),
    ("RET_TEST_029", "Pure water freezes at 0 degrees Celsius and boils at 100 degrees Celsius at 1 atm.", "General knowledge", "Under standard atmospheric pressure (1 atm), pure water freezes at 0 degrees Celsius and boils at 100 degrees Celsius."),
    ("RET_TEST_030", "The modern English alphabet consists of 26 letters.", "General knowledge", "The modern English alphabet contains 26 letters, consisting of five vowels (A, E, I, O, U) and 21 consonants."),
    ("RET_TEST_031", "The relational model for databases was proposed by Edgar F. Codd at IBM.", "Technology", "The relational model for database management was proposed by Edgar F. Codd in 1970 while working at IBM Research."),
    ("RET_TEST_032", "The Hubble Space Telescope was placed in orbit in April 1990.", "Science", "The Hubble Space Telescope was launched into low Earth orbit in April 1990 aboard the Space Shuttle Discovery."),
    ("RET_TEST_033", "The Suez Canal officially opened for navigation in November 1869.", "History", "The Suez Canal in Egypt officially opened on November 17, 1869, connecting the Mediterranean Sea to the Red Sea."),
    ("RET_TEST_034", "The Atacama Desert in Chile is the driest non-polar desert in the world.", "Geography", "The Atacama Desert in northern Chile is considered the driest non-polar desert in the world."),
    ("RET_TEST_035", "The Supreme Court of the United States includes a Chief Justice and eight associate justices.", "Politics and government", "The Supreme Court of the United States consists of the Chief Justice and eight associate justices appointed for life under good behavior."),
    ("RET_TEST_036", "The Federal Reserve System was established by the Federal Reserve Act in 1913.", "Economics", "The Federal Reserve is the central banking system of the United States, created on December 23, 1913 by the Federal Reserve Act."),
    ("RET_TEST_037", "James Naismith created basketball in Springfield, Massachusetts in 1891.", "Sports", "James Naismith invented the game of basketball in December 1891 in Springfield, Massachusetts, writing the original 13 rules."),
    ("RET_TEST_038", "Smallpox was officially certified as globally eradicated by the WHO in 1980.", "Medicine and health", "Smallpox was officially declared eradicated by the World Health Organization in May 1980 following a global vaccination campaign."),
    ("RET_TEST_039", "The IPCC was founded in 1988 by the WMO and UNEP.", "Environment", "The IPCC was created in 1988 by the World Meteorological Organization and the United Nations Environment Programme to provide scientific assessments on climate change."),
    ("RET_TEST_040", "The University of Bologna was founded in 1088.", "Education", "The University of Bologna in Italy, founded in 1088, is widely considered the oldest continuously operating university in the world."),
    ("RET_TEST_041", "Mars has two small natural moons named Phobos and Deimos.", "Space", "Mars has two small natural satellites, Phobos and Deimos, discovered by Asaph Hall in August 1877."),
    ("RET_TEST_042", "Apple Inc. was established in April 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.", "Companies and organizations", "Apple Inc. was founded on April 1, 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne to develop and sell the Apple I personal computer."),
    ("RET_TEST_043", "Ada Lovelace wrote the first algorithm intended for Charles Babbage's Analytical Engine.", "Famous people", "Ada Lovelace wrote the first algorithm intended to be carried out by Charles Babbage's mechanical Analytical Engine in 1843."),
    ("RET_TEST_044", "Impressionism was an artistic movement that arose in 19th-century France.", "Culture", "Impressionism was a 19th-century art movement in France characterized by visible brush strokes, open composition, and emphasis on accurate depiction of light."),
    ("RET_TEST_045", "Diamond registers a rating of 10 on the Mohs scale of mineral hardness.", "General knowledge", "Diamond is the hardest naturally occurring substance known, rating 10 on the Mohs scale of mineral hardness."),
    ("RET_TEST_046", "JavaScript was created by Brendan Eich in 1995.", "Technology", "JavaScript was created by Brendan Eich in 1995 while working at Netscape Communications Corporation."),
    ("RET_TEST_047", "The electron was discovered by J. J. Thomson in 1897.", "Science", "The electron was discovered in 1897 by J. J. Thomson during experiments with cathode ray tubes."),
    ("RET_TEST_048", "The Treaty of Versailles was signed on June 28, 1919.", "History", "The Treaty of Versailles was signed on June 28, 1919, officially ending World War I between Germany and the Allied Powers."),
    ("RET_TEST_049", "The Amazon River is the largest river by water discharge volume in the world.", "Geography", "The Amazon River in South America is the largest river by discharge volume of water in the world and the second longest after the Nile."),
    ("RET_TEST_050", "The Bretton Woods Conference established the IMF and World Bank in 1944.", "Economics", "The Bretton Woods Conference in July 1944 established the International Monetary Fund and the International Bank for Reconstruction and Development."),
]

retrieval_test_rows = []
for qid, qtext, topic, gold_text in TEST_QUERIES:
    # 1 Gold passage
    retrieval_test_rows.append({
        "claim_id": qid,
        "claim_text": qtext,
        "candidate_passage_id": f"{qid}_GOLD",
        "candidate_passage_text": gold_text,
        "relevance_label": "1",
        "evidence_role": "GOLD_SUPPORT",
    })
    # 2 Topical / partial passages
    topic_pool = passage_by_topic.get(topic, passage_pool)
    for role_name in ("PARTIAL", "TOPICALLY_RELATED"):
        p = random.choice(topic_pool)
        retrieval_test_rows.append({
            "claim_id": qid,
            "claim_text": qtext,
            "candidate_passage_id": p["passage_id"],
            "candidate_passage_text": p["passage_text"],
            "relevance_label": "0",
            "evidence_role": role_name,
        })
    # 7 Irrelevant passages from diverse topics
    other_topics = [t for t in TOPIC_PASSAGES if t != topic]
    for _ in range(7):
        rand_t = random.choice(other_topics)
        rand_p = random.choice(passage_by_topic[rand_t])
        retrieval_test_rows.append({
            "claim_id": qid,
            "claim_text": qtext,
            "candidate_passage_id": rand_p["passage_id"],
            "candidate_passage_text": rand_p["passage_text"],
            "relevance_label": "0",
            "evidence_role": "IRRELEVANT",
        })

# ── 4. Write Files ──────────────────────────────────────────────────────────
# 4.1 passage_pool.jsonl
pool_path = os.path.join(OUT_DIR, "passage_pool.jsonl")
with open(pool_path, "w", encoding="utf-8") as f:
    for p in passage_pool:
        f.write(json.dumps(p, ensure_ascii=True) + "\n")
print(f"Wrote {len(passage_pool)} passages to {pool_path}")

# 4.2 claim_candidates.csv
cands_path = os.path.join(OUT_DIR, "claim_candidates.csv")
with open(cands_path, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["claim_id", "claim_text", "candidate_passage_id", "candidate_passage_text", "relevance_label", "evidence_role"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(claim_candidates_rows)
print(f"Wrote {len(claim_candidates_rows)} candidate rows to {cands_path}")

# 4.3 retrieval_test.csv
test_path = os.path.join(OUT_DIR, "retrieval_test.csv")
with open(test_path, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["claim_id", "claim_text", "candidate_passage_id", "candidate_passage_text", "relevance_label", "evidence_role"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(retrieval_test_rows)
print(f"Wrote {len(retrieval_test_rows)} test candidate rows to {test_path}")

print("\nLayer 2 Retrieval dataset generation complete.")
