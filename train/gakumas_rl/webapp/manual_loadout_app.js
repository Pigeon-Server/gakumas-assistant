/* global window */
(function () {
  'use strict';

  const catalog = window.GAKUMAS_MANUAL_LOADOUT_CATALOG || {
    scenarios: [],
    idols: [],
    cards: [],
    card_groups: [],
    drinks: [],
    items: [],
  };

  const effectLabels = {
    Score: '打分',
    Review: '好印象',
    Block: '元气',
    Aggressive: '干劲',
    Concentration: '强气',
    FullPowerPoint: '全力',
    ParameterBuff: '好调',
    LessonBuff: '集中',
    Preservation: '温存',
    OverPreservation: '余温存',
    Enthusiastic: '热意',
    Sleepy: '眠气',
    Panic: '恐慌',
  };

  const stageLabels = {
    '': '未指定',
    ProduceStepType_AuditionMid1: '中间考核 1',
    ProduceStepType_AuditionMid2: '中间考核 2',
    ProduceStepType_AuditionFinal: '最终考核',
  };

  function normalizedText(value) {
    return String(value || '')
      .toLowerCase()
      .replace(/[\s\-_]+/g, '')
      .trim();
  }

  function safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function buildCardGroups(cards) {
    const groupsById = new Map();
    safeArray(cards).forEach((card) => {
      const cardId = String(card && card.card_id ? card.card_id : '');
      if (!cardId) {
        return;
      }
      let group = groupsById.get(cardId);
      if (!group) {
        group = {
          key: cardId,
          card_id: cardId,
          name: String(card.name || cardId),
          raw_name: String(card.base_raw_name || card.raw_name || cardId),
          category: String(card.category || ''),
          rarity: String(card.rarity || ''),
          plan_type: String(card.plan_type || ''),
          effect_types: [],
          description: String(card.description || ''),
          variants: [],
        };
        groupsById.set(cardId, group);
      }
      group.variants.push(card);
      safeArray(card.effect_types).forEach((effectType) => {
        if (!group.effect_types.includes(effectType)) {
          group.effect_types.push(effectType);
        }
      });
    });
    return Array.from(groupsById.values())
      .map((group) => {
        group.variants.sort((left, right) => Number(left.upgrade_count || 0) - Number(right.upgrade_count || 0));
        return group;
      })
      .sort((left, right) => `${left.name}-${left.card_id}`.localeCompare(`${right.name}-${right.card_id}`, 'zh-CN'));
  }

  const cardGroups = safeArray(catalog.card_groups).length ? safeArray(catalog.card_groups) : buildCardGroups(catalog.cards);
  catalog.card_groups = cardGroups;

  function findCard(cardId, upgradeCount) {
    return catalog.cards.find(
      (item) => item.card_id === cardId && Number(item.upgrade_count || 0) === Number(upgradeCount || 0),
    );
  }

  function findDrink(drinkId) {
    return catalog.drinks.find((item) => item.id === drinkId);
  }

  function findItem(itemId) {
    return catalog.items.find((item) => item.id === itemId);
  }

  function makeEmptyRecord() {
    const firstScenario = catalog.scenarios[0] || { alias: 'nia_master', stages: [] };
    return {
      label: '',
      scenario: firstScenario.alias,
      idol_card_id: '',
      stage_type: firstScenario.stages[0] || '',
      notes: '',
      deck: [],
      drinks: [],
      produce_items: [],
    };
  }

  function hydrateRecord(rawRecord) {
    const base = makeEmptyRecord();
    const deckEntries = safeArray(rawRecord.deck || rawRecord.cards).map((entry) => {
      const cardId = typeof entry === 'string' ? entry : String(entry.card_id || entry.id || '');
      const upgradeCount = typeof entry === 'string' ? 0 : Number(entry.upgrade_count || entry.upgradeCount || 0);
      const count = typeof entry === 'string' ? 1 : Math.max(1, Number(entry.count || 1));
      const card = findCard(cardId, upgradeCount) || {
        card_id: cardId,
        upgrade_count: upgradeCount,
        name: cardId,
        raw_name: cardId,
        rarity: '',
        category: '',
        effect_types: [],
        image_path: '',
      };
      return {
        card_id: card.card_id,
        upgrade_count: Number(card.upgrade_count || 0),
        count,
        name: card.name,
        raw_name: card.raw_name,
        rarity: card.rarity,
        category: card.category,
        effect_types: safeArray(card.effect_types),
        image_path: card.image_path,
      };
    });
    const drinks = safeArray(rawRecord.drinks).map((drinkId) => {
      const drink = findDrink(String(drinkId || '')) || {
        id: String(drinkId || ''),
        name: String(drinkId || ''),
        raw_name: String(drinkId || ''),
        rarity: '',
        effect_types: [],
      };
      return {
        id: drink.id,
        name: drink.name,
        raw_name: drink.raw_name,
        rarity: drink.rarity,
        effect_types: safeArray(drink.effect_types),
      };
    });
    const produceItems = safeArray(rawRecord.produce_items || rawRecord.items).map((itemId) => {
      const item = findItem(String(itemId || '')) || {
        id: String(itemId || ''),
        name: String(itemId || ''),
        raw_name: String(itemId || ''),
        is_exam_effect: false,
      };
      return {
        id: item.id,
        name: item.name,
        raw_name: item.raw_name,
        is_exam_effect: Boolean(item.is_exam_effect),
      };
    });
    return {
      label: String(rawRecord.label || rawRecord.name || base.label),
      scenario: String(rawRecord.scenario || base.scenario),
      idol_card_id: String(rawRecord.idol_card_id || base.idol_card_id),
      stage_type: String(rawRecord.stage_type || base.stage_type),
      notes: String(rawRecord.notes || ''),
      deck: deckEntries,
      drinks,
      produce_items: produceItems,
    };
  }

  function normalizeForExport(record) {
    return {
      label: String(record.label || '').trim(),
      scenario: String(record.scenario || '').trim(),
      idol_card_id: String(record.idol_card_id || '').trim(),
      stage_type: String(record.stage_type || '').trim(),
      notes: String(record.notes || '').trim(),
      deck: safeArray(record.deck).map((entry) => ({
        card_id: entry.card_id,
        upgrade_count: Number(entry.upgrade_count || 0),
        count: Math.max(1, Number(entry.count || 1)),
      })),
      drinks: safeArray(record.drinks).map((entry) => entry.id),
      produce_items: safeArray(record.produce_items).map((entry) => entry.id),
    };
  }

  function downloadText(filename, payload) {
    const blob = new Blob([payload], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  window.manualLoadoutApp = function manualLoadoutApp() {
    return {
      catalog,
      filters: {
        cardSearch: '',
        cardEffect: '',
        drinkSearch: '',
        itemSearch: '',
        idolSearch: '',
      },
      selectedCardUpgrades: {},
      current: makeEmptyRecord(),
      records: [],
      editingIndex: -1,
      importError: '',

      init() {
        this.ensureCardSelections();
        this.ensureStageValid();
      },

      effectLabel(effectType) {
        return effectLabels[effectType] || effectType;
      },

      stageLabel(stageType) {
        return stageLabels[stageType] || stageType || '未指定';
      },

      cardDisplayName(card) {
        if (!card) {
          return '';
        }
        return Number(card.upgrade_count || 0) > 0 ? `${card.name} +${card.upgrade_count}` : card.name;
      },

      ensureCardSelections() {
        safeArray(catalog.card_groups).forEach((group) => {
          if (this.selectedCardUpgrades[group.card_id] !== undefined) {
            return;
          }
          const firstVariant = safeArray(group.variants)[0];
          this.selectedCardUpgrades[group.card_id] = Number(firstVariant && firstVariant.upgrade_count ? firstVariant.upgrade_count : 0);
        });
      },

      selectedCardVariant(group) {
        const variants = safeArray(group && group.variants);
        if (!variants.length) {
          return null;
        }
        const requestedUpgrade = Number(this.selectedCardUpgrades[group.card_id] || 0);
        return variants.find((item) => Number(item.upgrade_count || 0) === requestedUpgrade) || variants[0];
      },

      idolChoices() {
        const query = normalizedText(this.filters.idolSearch);
        return catalog.idols
          .filter((item) => {
            if (!query) {
              return true;
            }
            const haystack = normalizedText(
              [item.id, item.name, item.character_id, item.plan_type, item.exam_effect_type, item.rarity].join(' '),
            );
            return haystack.includes(query);
          })
          .slice(0, 180);
      },

      availableEffectTypes() {
        return Object.keys(effectLabels);
      },

      stageChoices() {
        const scenario = catalog.scenarios.find((item) => item.alias === this.current.scenario);
        const choices = scenario ? scenario.stages : [];
        return [''].concat(choices);
      },

      ensureStageValid() {
        const choices = this.stageChoices();
        if (!choices.includes(this.current.stage_type)) {
          this.current.stage_type = choices[0] || '';
        }
      },

      filteredCards() {
        const query = normalizedText(this.filters.cardSearch);
        const effect = String(this.filters.cardEffect || '').trim();
        return safeArray(catalog.card_groups)
          .filter((group) => {
            if (effect && !safeArray(group.effect_types).includes(effect)) {
              return false;
            }
            if (!query) {
              return true;
            }
            const variantNames = safeArray(group.variants)
              .map((item) => [item.raw_name, item.upgrade_count, item.key].join(' '))
              .join(' ');
            const haystack = normalizedText(
              [
                group.card_id,
                group.name,
                group.raw_name,
                group.rarity,
                group.category,
                group.plan_type,
                safeArray(group.effect_types).join(' '),
                group.description,
                variantNames,
              ].join(' '),
            );
            return haystack.includes(query);
          })
          .slice(0, 96);
      },

      filteredDrinks() {
        const query = normalizedText(this.filters.drinkSearch);
        return catalog.drinks
          .filter((item) => {
            if (!query) {
              return true;
            }
            const haystack = normalizedText(
              [item.id, item.name, item.raw_name, item.rarity, safeArray(item.effect_types).join(' '), item.description].join(' '),
            );
            return haystack.includes(query);
          })
          .slice(0, 80);
      },

      filteredItems() {
        const query = normalizedText(this.filters.itemSearch);
        return catalog.items
          .filter((item) => {
            if (!query) {
              return true;
            }
            const haystack = normalizedText(
              [item.id, item.name, item.raw_name, item.description, item.is_exam_effect ? 'exam' : '', item.is_challenge ? 'challenge' : ''].join(' '),
            );
            return haystack.includes(query);
          })
          .slice(0, 80);
      },

      currentDeckCount() {
        return safeArray(this.current.deck).reduce((sum, entry) => sum + Math.max(1, Number(entry.count || 1)), 0);
      },

      currentDrinkCount() {
        return safeArray(this.current.drinks).length;
      },

      currentItemCount() {
        return safeArray(this.current.produce_items).length;
      },

      addDeckCardGroup(group, amount = 1) {
        const variant = this.selectedCardVariant(group);
        if (!variant) {
          return;
        }
        this.addDeckCard(variant, amount);
      },

      addDeckCard(card, amount = 1) {
        const existing = this.current.deck.find(
          (entry) => entry.card_id === card.card_id && Number(entry.upgrade_count || 0) === Number(card.upgrade_count || 0),
        );
        if (existing) {
          existing.count += amount;
        } else {
          this.current.deck.push({
            card_id: card.card_id,
            upgrade_count: Number(card.upgrade_count || 0),
            count: amount,
            name: card.name,
            raw_name: card.raw_name,
            rarity: card.rarity,
            category: card.category,
            effect_types: safeArray(card.effect_types),
            image_path: card.image_path,
          });
        }
        this.current.deck.sort((left, right) => {
          const leftKey = `${left.name}-${left.upgrade_count}`;
          const rightKey = `${right.name}-${right.upgrade_count}`;
          return leftKey.localeCompare(rightKey, 'zh-CN');
        });
      },

      changeDeckCount(entry, delta) {
        entry.count = Math.max(0, Number(entry.count || 0) + Number(delta || 0));
        this.current.deck = this.current.deck.filter((item) => Number(item.count || 0) > 0);
      },

      addDrink(drink) {
        this.current.drinks.push({
          id: drink.id,
          name: drink.name,
          raw_name: drink.raw_name,
          rarity: drink.rarity,
          effect_types: safeArray(drink.effect_types),
        });
      },

      addProduceItem(item) {
        this.current.produce_items.push({
          id: item.id,
          name: item.name,
          raw_name: item.raw_name,
          is_exam_effect: Boolean(item.is_exam_effect),
        });
      },

      removeDeckEntry(index) {
        this.current.deck.splice(index, 1);
      },

      removeListEntry(fieldName, index) {
        this.current[fieldName].splice(index, 1);
      },

      resetCurrent() {
        this.current = makeEmptyRecord();
        this.editingIndex = -1;
        this.importError = '';
        this.ensureStageValid();
      },

      saveCurrentRecord() {
        if (!this.current.deck.length) {
          window.alert('当前记录还没有手牌列表。');
          return;
        }
        const payload = normalizeForExport(this.current);
        if (this.editingIndex >= 0) {
          this.records.splice(this.editingIndex, 1, payload);
        } else {
          this.records.push(payload);
        }
        this.resetCurrent();
      },

      loadRecord(index) {
        this.current = hydrateRecord(this.records[index]);
        this.editingIndex = index;
        this.ensureStageValid();
      },

      duplicateRecord(index) {
        const copy = deepClone(this.records[index]);
        copy.label = copy.label ? `${copy.label} copy` : 'copy';
        this.records.splice(index + 1, 0, copy);
      },

      deleteRecord(index) {
        this.records.splice(index, 1);
        if (this.editingIndex === index) {
          this.resetCurrent();
        } else if (this.editingIndex > index) {
          this.editingIndex -= 1;
        }
      },

      exportRecords() {
        const exportable = this.records.length ? this.records : (this.current.deck.length ? [normalizeForExport(this.current)] : []);
        if (!exportable.length) {
          window.alert('没有可导出的记录。');
          return;
        }
        const payload = exportable.map((record) => JSON.stringify(record)).join('\n') + '\n';
        downloadText(`manual_exam_setup_${Date.now()}.jsonl`, payload);
      },

      previewJsonl() {
        const exportable = this.records.length ? this.records : (this.current.deck.length ? [normalizeForExport(this.current)] : []);
        return exportable.map((record) => JSON.stringify(record)).join('\n');
      },

      importJsonl(event) {
        const file = event.target.files && event.target.files[0];
        if (!file) {
          return;
        }
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const parsedRecords = String(reader.result || '')
              .split(/\r?\n/)
              .map((line) => line.trim())
              .filter(Boolean)
              .map((line) => normalizeForExport(hydrateRecord(JSON.parse(line))));
            this.records = parsedRecords;
            this.importError = '';
          } catch (error) {
            this.importError = error instanceof Error ? error.message : String(error);
          }
        };
        reader.readAsText(file);
      },
    };
  };
})();
