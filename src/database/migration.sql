-- CSV verilerini MEVCUT topics tablosuna ekle
-- Schema zaten var, sadece eksik kolonları ekleyip veri import edeceğiz

-- ADIM 1: Eksik kolonları ekle
ALTER TABLE topics ADD COLUMN keywords TEXT;
ALTER TABLE topics ADD COLUMN historical_period TEXT;

-- ADIM 2: CSV verilerini mevcut topics tablosuna ekle
INSERT INTO topics (
    topic,
    description,
    keywords,
    historical_period,
    category,
    priority,
    target_duration_minutes,
    status,
    created_at,
    updated_at
) VALUES
('Pompeii''s Final Night', 'A slow, atmospheric narrative exploring the last peaceful night in Pompeii before the eruption of Mount Vesuvius.', 'Vesuvius, eruption, Roman Empire, Campania, tragedy', '1st Century CE', 'historical_narrative', 3, 120, 'completed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Library of Alexandria', 'The last quiet night before flames consumed one of history''s greatest knowledge centers.', 'books, scrolls, ancient knowledge, fire, Egypt', '3rd Century BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Fall of Constantinople', 'A solemn recounting of the final twilight hours inside the walls of a fading empire.', 'Byzantine, Ottomans, siege, Hagia Sophia', '15th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Cleopatra''s Final Night', 'A tranquil, shadowy narrative told from the queen''s chamber as history closes in.', 'Ptolemaic, Egypt, Rome, Julius Caesar, asp', '1st Century BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Day of Atlantis', 'An ethereal story of a civilization''s final sunset before vanishing beneath the waves.', 'mythology, ocean, civilization, Plato, mystery', 'Mythical', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Tutankhamun''s Burial Night', 'The quiet hours before the young pharaoh was sealed away for eternity in his golden tomb.', 'Egypt, pharaoh, tomb, gold, Valley of Kings', '14th Century BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Night Before Gettysburg', 'A reflective narrative capturing the stillness before one of history''s most pivotal battles.', 'Civil War, Pennsylvania, Lincoln, battlefield, Union', '19th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Leonardo''s Last Canvas', 'The final evening Leonardo da Vinci spent contemplating his life''s work in his Loire Valley home.', 'Renaissance, art, genius, France, Mona Lisa', '16th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Lighthouse of Alexandria''s Last Night', 'A gentle tale of the ancient world''s greatest beacon before earthquakes silenced its light forever.', 'Seven Wonders, navigation, Mediterranean, ancient engineering', '2nd Century CE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Caesar''s Final Senate Meeting', 'The twilight hours before Julius Caesar''s fateful walk to the Theatre of Pompey.', 'Roman Republic, assassination, Ides of March, Brutus', '1st Century BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Mammoth', 'A peaceful story of the final woolly mammoth as the ice age ended and forests replaced tundra.', 'Ice Age, extinction, climate change, Siberia, prehistoric', '10,000 BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Stonehenge''s Completion', 'The quiet satisfaction of ancient builders as they placed the final trilithon under starlight.', 'Neolithic, astronomy, ancient Britain, megaliths, druids', '2500 BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Great Wall''s Final Stone', 'A meditative narrative about the last worker to place a stone in China''s great fortification.', 'China, defense, Ming Dynasty, construction, sacrifice', '17th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Troy''s Final Watch', 'The last peaceful night on Troy''s walls before the Greeks emerged from their wooden horse.', 'Homer, Iliad, siege, Trojan Horse, ancient Greece', '12th Century BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Marco Polo''s Departure from Venice', 'The thoughtful evening before the young merchant embarked on his legendary journey to the East.', 'Silk Road, exploration, Venice, Kublai Khan, adventure', '13th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Dodo', 'A gentle story about the final moments of the dodo bird on the peaceful island of Mauritius.', 'extinction, Mauritius, flightless bird, Dutch colonization', '17th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Machu Picchu''s Abandonment', 'The final night in the cloud city before the Inca left their mountain sanctuary forever.', 'Inca, Peru, Andes, Spanish conquest, mountain citadel', '16th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Roman Legion', 'A contemplative tale of the final Roman soldiers withdrawing from Britain''s distant shores.', 'Roman Empire, Britain, withdrawal, Hadrian''s Wall, decline', '5th Century CE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Beethoven''s Ninth Symphony', 'The quiet hours before Ludwig van Beethoven premiered his final and greatest symphony.', 'classical music, Vienna, deafness, Ode to Joy, genius', '19th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Temple of Artemis'' Final Prayer', 'A serene narrative about the last ceremony held in one of the Seven Wonders of the Ancient World.', 'Seven Wonders, Ephesus, Greek goddess, ancient temple', '5th Century CE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Columbus'' Last Night in Spain', 'The reflective evening before Christopher Columbus departed on his voyage to the unknown West.', 'Age of Exploration, New World, Atlantic, Spanish crown', '15th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Pharaoh''s Dream', 'A dreamy narrative about Cleopatra VII''s final night as Egypt''s independence slipped away.', 'Ptolemaic dynasty, Roman conquest, Alexandria, Mark Antony', '1st Century BCE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Hadrian''s Wall Completion', 'The satisfied rest of Roman engineers after completing their monumental barrier across Britain.', 'Roman engineering, Hadrian, frontier, Scotland, construction', '2nd Century CE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Colosseum''s First Night', 'A calm story about the evening before Rome''s greatest amphitheater opened to the public.', 'Roman Empire, gladiators, architecture, entertainment, Flavian', '1st Century CE', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Shakespeare''s Final Performance', 'The quiet moments after William Shakespeare''s last appearance on the Globe Theatre stage.', 'Elizabethan theatre, Globe Theatre, literature, Renaissance', '17th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Knight of Camelot', 'A peaceful tale of the final knight keeping vigil as the age of chivalry drew to a close.', 'Arthurian legend, chivalry, medieval romance, Round Table', 'Medieval', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Newton''s Apple Tree Night', 'The tranquil evening Isaac Newton spent beneath his apple tree, contemplating gravity''s mysteries.', 'Scientific Revolution, physics, gravity, Cambridge, mathematics', '17th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Viking Voyage', 'A serene narrative about the final longboat to return from distant shores as the Viking age ended.', 'Norse, exploration, longships, Scandinavia, raids', '11th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('Galileo''s Telescope Night', 'The wonder-filled evening Galileo first turned his telescope toward Jupiter''s dancing moons.', 'Scientific Revolution, astronomy, telescope, Jupiter, discovery', '17th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
('The Last Samurai''s Meditation', 'A contemplative story about a samurai''s final night of reflection as the old ways gave way to new.', 'Meiji Restoration, bushido, honor, Japan, tradition', '19th Century', 'historical_narrative', 2, 120, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- ADIM 3: Yeni kolonlar için indeksler
CREATE INDEX IF NOT EXISTS idx_topics_keywords ON topics(keywords);
CREATE INDEX IF NOT EXISTS idx_topics_historical_period ON topics(historical_period);

-- ADIM 4: Kontrol sorguları
SELECT 'Migration tamamlandı!' as message;
SELECT COUNT(*) as toplam_kayit FROM topics;
SELECT COUNT(*) as historical_narrative_sayisi FROM topics WHERE category = 'historical_narrative';

-- ADIM 5: Örnek veriler
SELECT topic, keywords, historical_period, status
FROM topics
WHERE keywords LIKE '%Roman%' OR keywords LIKE '%Rome%'
LIMIT 5;

-- ADIM 6: Historical period grupları
SELECT historical_period, COUNT(*) as sayı
FROM topics
WHERE historical_period IS NOT NULL
GROUP BY historical_period
ORDER BY sayı DESC;