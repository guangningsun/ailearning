#!/usr/bin/env node
/**
 * Update content field in data.js by reading docs/zh.md files.
 * Maps phases/ paths to lessons/ paths and converts markdown to HTML.
 *
 * Run: node site/update-content.js
 */

const fs = require('fs');
const path = require('path');

const SITE_DIR = __dirname;
const REPO_ROOT = path.resolve(SITE_DIR, '..');
const DATA_PATH = path.join(SITE_DIR, 'data.js');

let marked;
try {
  marked = require('marked2');
} catch (e) {
  console.error('❌ marked2 not found. Run: npm install marked2');
  process.exit(1);
}

// Map phases/ path to lessons/ path
function mapPath(phasesPath) {
  return 'lessons/' + phasesPath.replace(/^phases\//, '');
}

// Extract content from docs/zh.md (fallback to docs/en.md)
function extractContent(relPath) {
  const lessonsRel = mapPath(relPath);
  const zhPath = path.join(REPO_ROOT, lessonsRel, 'docs', 'zh.md');
  const enPath = path.join(REPO_ROOT, lessonsRel, 'docs', 'en.md');

  let docPath = zhPath;
  if (!fs.existsSync(zhPath)) {
    if (fs.existsSync(enPath)) {
      docPath = enPath;
    } else {
      return null;
    }
  }

  try {
    const content = fs.readFileSync(docPath, 'utf8');
    return marked(content);
  } catch (e) {
    console.warn(`⚠️  Failed to read ${docPath}: ${e.message}`);
    return null;
  }
}

// Escape string for JSON
function jsonEscape(str) {
  return str
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}

// Find the closing quote position for an escaped JSON string
function findStringEnd(code, startPos) {
  let i = startPos;
  const len = code.length;
  while (i < len) {
    if (code[i] === '\\') {
      i += 2;
    } else if (code[i] === '"') {
      return i;
    } else {
      i++;
    }
  }
  return -1;
}

// Main update function
function updateContent() {
  console.log('📖 Loading data.js...');
  const code = fs.readFileSync(DATA_PATH, 'utf8');

  let updatedCount = 0;
  let skippedCount = 0;
  let errorCount = 0;

  // Pattern to find lesson URLs
  const lessonUrlPattern = /"url":\s*"lesson\.html\?path=phases\/([^"]+)"/g;

  // Phase 1: Scan all lessons and collect their content ranges (based on original `code`)
  const entries = [];

  let match;
  while ((match = lessonUrlPattern.exec(code)) !== null) {
    const pathPart = match[1];
    const urlEndPos = match.index + match[0].length;

    // Search for "content": " within next 3000 chars
    const searchArea = code.slice(urlEndPos, urlEndPos + 3000);
    const contentMatch = /"content":\s*"/.exec(searchArea);
    if (!contentMatch) {
      console.warn(`⚠️  No content field found: ${pathPart}`);
      errorCount++;
      continue;
    }

    const contentStartPos = urlEndPos + contentMatch.index + contentMatch[0].length;
    const contentEndPos = findStringEnd(code, contentStartPos);
    if (contentEndPos === -1) {
      console.warn(`⚠️  Could not find closing quote: ${pathPart}`);
      errorCount++;
      continue;
    }

    entries.push({
      pathPart,
      contentStartPos,
      contentEndPos,
      currentContent: code.slice(contentStartPos, contentEndPos),
    });
  }

  // Phase 2: Process each entry, building result with cumulative offset tracking
  let result = code;
  let offset = 0; // cumulative shift from replacements

  for (const entry of entries) {
    const { pathPart, currentContent } = entry;
    // Adjust positions for cumulative offset
    const contentStartPos = entry.contentStartPos + offset;
    const contentEndPos = entry.contentEndPos + offset;

    // Unescape current content for comparison
    const currentUnescaped = currentContent
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\');

    // Extract new content
    const newHtmlContent = extractContent(pathPart);

    if (newHtmlContent === null) {
      skippedCount++;
      continue;
    }

    // Check if current content is already Chinese (and new content is English)
    const isChineseCurrent = /[\u4e00-\u9fa5]/.test(currentUnescaped.slice(0, 200));
    const isChineseNew = /[\u4e00-\u9fa5]/.test(newHtmlContent.slice(0, 200));

    // If current is Chinese but new is English, skip (keep Chinese)
    if (isChineseCurrent && !isChineseNew) {
      console.log(`  ⏭️  Skipping (already Chinese): ${pathPart}`);
      skippedCount++;
      continue;
    }

    if (!isChineseCurrent && isChineseNew) {
      console.log(`  🔄 Updating English to Chinese: ${pathPart}`);
    } else if (currentUnescaped.slice(0, 100) === newHtmlContent.slice(0, 100)) {
      skippedCount++;
      continue;
    } else {
      console.log(`  🔄 Updating content: ${pathPart}`);
    }

    // Escape new content for JSON
    const escapedNewContent = jsonEscape(newHtmlContent);

    // Replace using adjusted positions (no regex, no search — we know exactly where it is)
    const newContentStr = '"' + escapedNewContent + '"';
    const oldLen = contentEndPos - contentStartPos + 2; // +2 for quotes
    result = result.slice(0, contentStartPos - 1) + newContentStr + result.slice(contentEndPos + 1);
    const newLen = newContentStr.length;
    offset += newLen - oldLen;
    updatedCount++;
    console.log(`    ✅ Done: ${pathPart}`);
  }

  // Write updated data.js
  fs.writeFileSync(DATA_PATH, result, 'utf8');

  console.log(`\n📊 Results:`);
  console.log(`   Updated: ${updatedCount}`);
  console.log(`   Skipped: ${skippedCount}`);
  console.log(`   Errors: ${errorCount}`);

  if (updatedCount > 0) {
    console.log(`\n✅ Updated data.js`);
  } else {
    console.log(`\n⚠️  No changes made`);
  }
}

// Run
updateContent();