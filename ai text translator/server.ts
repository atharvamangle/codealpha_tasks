import express, { Request, Response } from 'express';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.use(express.json({ limit: '1mb' }));

// Helper to execute generation with retries for temporary model demand
async function generateWithRetry(ai: GoogleGenAI, params: any, maxRetries = 2): Promise<any> {
  let lastError: any;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await ai.models.generateContent(params);
    } catch (err: any) {
      lastError = err;
      const errStr = String(err?.message || '');
      const isTransient =
        errStr.includes('503') ||
        errStr.includes('high demand') ||
        errStr.includes('UNAVAILABLE') ||
        errStr.includes('RESOURCE_EXHAUSTED') ||
        errStr.includes('429');

      if (isTransient && attempt < maxRetries) {
        // Wait 1.2s before retry
        await new Promise((resolve) => setTimeout(resolve, 1200 * (attempt + 1)));
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

// Format errors nicely
function formatErrorMessage(error: any): string {
  if (!error) return 'An unexpected error occurred.';
  const msg = error.message || String(error);
  if (msg.includes('high demand') || msg.includes('503')) {
    return 'The AI translation service is currently experiencing high demand. Please try clicking Translate again in a few seconds.';
  }
  if (msg.includes('GEMINI_API_KEY') || msg.includes('API key')) {
    return 'Gemini API key is missing or invalid. Please check your environment variables or AI Studio Secrets.';
  }
  if (msg.includes('RESOURCE_EXHAUSTED') || msg.includes('429')) {
    return 'Rate limit reached. Please wait a moment before submitting another translation.';
  }
  try {
    const jsonMatch = msg.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      if (parsed?.error?.message) {
        return parsed.error.message;
      }
    }
  } catch {
    // Ignore JSON parsing errors
  }
  return msg;
}

// Helper to initialize Gemini
function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey.trim() === '' || apiKey === 'MY_GEMINI_API_KEY') {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Health check endpoint
app.get('/api/health', (_req: Request, res: Response) => {
  const apiKey = process.env.GEMINI_API_KEY;
  const hasKey = Boolean(apiKey && apiKey.trim() !== '' && apiKey !== 'MY_GEMINI_API_KEY');
  res.json({
    status: 'ok',
    hasKey,
    model: 'gemini-3.8-flash',
    timestamp: new Date().toISOString(),
  });
});

// Detect language endpoint
app.post('/api/detect', async (req: Request, res: Response) => {
  try {
    const { text } = req.body;
    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      res.status(400).json({ error: 'Text is required for language detection.' });
      return;
    }

    const ai = getGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'Missing GEMINI_API_KEY. Please configure your API key in environment variables or AI Studio Secrets.',
      });
      return;
    }

    const prompt = `Identify the natural language of the following text snippet. Return the result in JSON format with fields: languageCode (ISO 639-1 code like en, mr, hi, fr, es, de, ja, etc.) and languageName (in English).

Text snippet:
"""
${text.slice(0, 1000)}
"""`;

    const response = await generateWithRetry(ai, {
      model: 'gemini-3.8-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            languageCode: { type: Type.STRING },
            languageName: { type: Type.STRING },
            confidence: { type: Type.STRING },
          },
          required: ['languageCode', 'languageName'],
        },
      },
    });

    const responseText = response.text || '{}';
    const parsed = JSON.parse(responseText);
    res.json(parsed);
  } catch (error: any) {
    console.error('Detection error:', error);
    res.status(500).json({
      error: formatErrorMessage(error),
    });
  }
});

// Translation endpoint
app.post('/api/translate', async (req: Request, res: Response) => {
  try {
    const { text, sourceLang = 'auto', targetLang = 'en', tone = 'natural' } = req.body;

    if (!text || typeof text !== 'string' || text.trim().length === 0) {
      res.status(400).json({ error: 'Please enter text to translate.' });
      return;
    }

    if (text.length > 5000) {
      res.status(400).json({ error: 'Text exceeds the 5,000 character limit. Please shorten your input.' });
      return;
    }

    const ai = getGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'Missing GEMINI_API_KEY in environment variables. Please check your AI Studio Secrets or .env file to enable translation services.',
        needsKey: true,
      });
      return;
    }

    const toneInstruction = {
      formal: 'Use an official, polite, and formal tone suitable for professional correspondence.',
      casual: 'Use an informal, relaxed, conversational everyday tone.',
      business: 'Use crisp, professional business terminology suited for enterprise meetings and emails.',
      natural: 'Use clear, natural, and idiomatic phrasing as spoken by native speakers.',
    }[tone as string] || 'Use natural, idiomatic phrasing.';

    const systemInstruction = `You are LinguaBridge AI, an expert high-accuracy multilingual translator.
Your goal is to provide exceptional, accurate, and culturally appropriate translations.
You support all global languages including English, Marathi, Hindi, Spanish, French, German, Japanese, Korean, Chinese, Arabic, Portuguese, Italian, and many others.
Translate accurately while preserving original context, tone, formatting, and punctuality.
Tone requirement: ${toneInstruction}
If source language is "auto", automatically identify the source language accurately.
If the target language or source language uses a non-Latin script (such as Devanagari for Marathi/Hindi, Kanji/Kana for Japanese, Hangul for Korean, Arabic script, Chinese characters, Cyrillic), provide Romanized phonetic transliteration/pronunciation guide to help users pronounce the translation.
Provide 1 or 2 alternative phrasings when useful, and a brief grammar or cultural nuance note if applicable.`;

    const userPrompt = `Translate the following text.
Source Language Specification: ${sourceLang === 'auto' ? 'Auto-Detect' : sourceLang}
Target Language Specification: ${targetLang}
Desired Tone: ${tone}

Original Text to translate:
"""
${text}
"""`;

    const response = await generateWithRetry(ai, {
      model: 'gemini-3.8-flash',
      contents: userPrompt,
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            translatedText: {
              type: Type.STRING,
              description: 'The translated text in the target language',
            },
            detectedSourceLanguageName: {
              type: Type.STRING,
              description: 'English name of the detected or specified source language (e.g. Marathi, English, Hindi)',
            },
            detectedSourceLanguageCode: {
              type: Type.STRING,
              description: 'ISO code of the source language (e.g. mr, hi, en, ja, fr, es)',
            },
            targetLanguageName: {
              type: Type.STRING,
              description: 'English name of the target language',
            },
            targetLanguageCode: {
              type: Type.STRING,
              description: 'ISO code of the target language',
            },
            phoneticTransliteration: {
              type: Type.STRING,
              description: 'Romanized phonetic guide or pronunciation (especially for non-Latin scripts like Marathi, Hindi, Japanese, Arabic, Korean, Chinese). If target is already Latin alphabet, provide phonetic guide only if helpful, else empty string.',
            },
            alternativePhrasings: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: '1 to 2 alternative natural translations or synonym expressions',
            },
            contextOrGrammarNote: {
              type: Type.STRING,
              description: 'Helpful nuance note regarding formality, gender, or cultural usage, if relevant; otherwise empty string',
            },
          },
          required: [
            'translatedText',
            'detectedSourceLanguageName',
            'detectedSourceLanguageCode',
            'targetLanguageName',
            'targetLanguageCode',
          ],
        },
      },
    });

    const responseText = response.text;
    if (!responseText) {
      throw new Error('Received an empty response from the translation engine.');
    }

    const data = JSON.parse(responseText);

    res.json({
      success: true,
      translatedText: data.translatedText,
      detectedSourceLanguageName: data.detectedSourceLanguageName,
      detectedSourceLanguageCode: data.detectedSourceLanguageCode,
      targetLanguageName: data.targetLanguageName,
      targetLanguageCode: data.targetLanguageCode,
      phoneticTransliteration: data.phoneticTransliteration || null,
      alternativePhrasings: data.alternativePhrasings || [],
      contextOrGrammarNote: data.contextOrGrammarNote || null,
      charCount: data.translatedText?.length || 0,
      wordCount: (data.translatedText || '').trim().split(/\s+/).filter(Boolean).length,
    });
  } catch (error: any) {
    console.error('Translation error:', error);
    res.status(500).json({
      error: formatErrorMessage(error),
    });
  }
});

// Vite middleware in dev or static files in production
async function startServer() {
  const isProduction = process.env.NODE_ENV === 'production';

  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`LinguaBridge AI server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
