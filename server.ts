import dotenv from 'dotenv';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express, { type Request, type Response, type NextFunction } from 'express';
import bcrypt from 'bcryptjs';
import db from './server/db.ts';
import {
  generateChatbotReply,
  generateResearchExplanation,
  isGeminiConfigured,
  MissingGeminiKeyError,
} from './server/gemini.ts';

if (fs.existsSync(path.resolve(process.cwd(), '.env'))) {
  dotenv.config();
}

const PORT = Number(process.env.PORT) || 3000;
const SESSION_COOKIE = 'np_session';
const SESSION_SECRET =
  process.env.SESSION_SECRET ||
  crypto.createHash('sha256').update(process.cwd() + '_neuropredict_local_secret').digest('hex');

interface SessionPayload {
  userId: number;
  email: string;
  exp: number;
}

function signSession(payload: SessionPayload): string {
  const encoded = Buffer.from(JSON.stringify(payload)).toString('base64url');
  const sig = crypto.createHmac('sha256', SESSION_SECRET).update(encoded).digest('base64url');
  return `${encoded}.${sig}`;
}

function verifySession(token: string | undefined): SessionPayload | null {
  if (!token || !token.includes('.')) return null;
  const [encoded, sig] = token.split('.');
  if (!encoded || !sig) return null;
  const expectedSig = crypto.createHmac('sha256', SESSION_SECRET).update(encoded).digest('base64url');
  const a = Buffer.from(sig);
  const b = Buffer.from(expectedSig);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  try {
    const parsed = JSON.parse(Buffer.from(encoded, 'base64url').toString('utf8')) as SessionPayload;
    if (!parsed.userId || !parsed.exp || Date.now() > parsed.exp) return null;
    return parsed;
  } catch {
    return null;
  }
}

function parseCookies(cookieHeader: string | undefined): Record<string, string> {
  const result: Record<string, string> = {};
  if (!cookieHeader) return result;
  for (const part of cookieHeader.split(';')) {
    const idx = part.indexOf('=');
    if (idx > 0) {
      const k = part.slice(0, idx).trim();
      const v = decodeURIComponent(part.slice(idx + 1).trim());
      result[k] = v;
    }
  }
  return result;
}

function isHttpsRequest(req: Request): boolean {
  const forwardedProto = req.headers['x-forwarded-proto'];
  const proto = Array.isArray(forwardedProto) ? forwardedProto[0] : forwardedProto;
  const origin = req.headers.origin;
  const referer = req.headers.referer;
  return Boolean(
    req.secure ||
      (proto && proto.split(',')[0].trim() === 'https') ||
      (typeof origin === 'string' && origin.startsWith('https://')) ||
      (typeof referer === 'string' && referer.startsWith('https://'))
  );
}

function setSessionCookie(
  req: Request,
  res: Response,
  userId: number,
  email: string,
  rememberMe = true
): string {
  const ttlMs = rememberMe ? 7 * 24 * 60 * 60 * 1000 : 12 * 60 * 60 * 1000;
  const token = signSession({ userId, email, exp: Date.now() + ttlMs });
  const isHttps = isHttpsRequest(req);
  res.cookie(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: isHttps ? 'none' : 'lax',
    secure: isHttps,
    partitioned: isHttps,
    path: '/',
    maxAge: ttlMs,
  });
  return token;
}

function extractSessionTokens(req: Request): string[] {
  const tokens: string[] = [];
  const authHeader = req.headers.authorization;
  if (typeof authHeader === 'string') {
    const match = authHeader.match(/^Bearer\s+(.+)$/i);
    if (match?.[1]?.trim()) {
      tokens.push(match[1].trim());
    }
  }
  const cookies = parseCookies(req.headers.cookie);
  if (cookies[SESSION_COOKIE]) {
    tokens.push(cookies[SESSION_COOKIE]);
  }
  return tokens;
}

function resolveValidSession(req: Request): { session: SessionPayload; token: string } | null {
  for (const token of extractSessionTokens(req)) {
    const session = verifySession(token);
    if (session) {
      return { session, token };
    }
  }
  return null;
}

interface AuthedRequest extends Request {
  user?: {
    id: number;
    fullName: string;
    email: string;
    role: string;
    institution: string;
    createdAt: string;
  };
}

function requireAuth(req: AuthedRequest, res: Response, next: NextFunction) {
  const resolved = resolveValidSession(req);
  if (!resolved) {
    res.status(401).json({ error: 'Authentication required. Please sign in.' });
    return;
  }
  const { session } = resolved;
  const row = db
    .prepare('SELECT id, full_name, email, role, institution, created_at FROM users WHERE id = ?')
    .get(session.userId) as
    | {
        id: number;
        full_name: string;
        email: string;
        role: string;
        institution: string;
        created_at: string;
      }
    | undefined;

  if (!row) {
    res.status(401).json({ error: 'User session is no longer valid.' });
    return;
  }

  req.user = {
    id: row.id,
    fullName: row.full_name,
    email: row.email,
    role: row.role,
    institution: row.institution,
    createdAt: row.created_at,
  };
  next();
}

function mapPredictionRow(row: any) {
  return {
    id: row.id,
    userId: row.user_id,
    analysisType: row.analysis_type,
    imageFilename: row.image_filename,
    imagePreviewData: row.image_preview_data,
    preprocessedPreviewData: row.preprocessed_preview_data,
    mriMetrics: row.mri_metrics_json ? JSON.parse(row.mri_metrics_json) : null,
    clinicalData: row.clinical_data_json ? JSON.parse(row.clinical_data_json) : null,
    clinicalPreprocessed: row.clinical_preprocessed_json
      ? JSON.parse(row.clinical_preprocessed_json)
      : null,
    predictedCategory: row.predicted_category,
    modelStatus: row.model_status,
    confidenceNote: row.confidence_note,
    summaryText: row.summary_text,
    geminiExplanation: row.gemini_explanation,
    createdAt: row.created_at,
  };
}

async function startServer() {
  const app = express();
  app.set('trust proxy', true);
  app.use(express.json({ limit: '15mb' }));
  app.use('/api', (_req, res, next) => {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
    next();
  });

  app.get('/api/health', (_req, res) => {
    res.json({
      ok: true,
      service: 'NeuroPredict AI Backend',
      geminiConfigured: isGeminiConfigured(),
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/api/sample-report.pdf', (_req, res) => {
    const samplePdfPath = path.resolve(
      process.cwd(),
      '.aistudio/artifacts/computer_use/NeuroPredict_Medical_Report.pdf'
    );
    if (!fs.existsSync(samplePdfPath)) {
      res.status(404).json({ error: 'Sample PDF report not found.' });
      return;
    }
    res.download(samplePdfPath, 'NeuroPredict_Medical_Report.pdf');
  });

  // --- Authentication Routes ---
  app.post('/api/auth/register', (req, res) => {
    const { fullName, email, password } = req.body ?? {};
    if (!fullName || typeof fullName !== 'string' || fullName.trim().length < 2) {
      res.status(400).json({ error: 'Full name must be at least 2 characters.' });
      return;
    }
    if (!email || typeof email !== 'string' || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())) {
      res.status(400).json({ error: 'Please provide a valid email address.' });
      return;
    }
    if (!password || typeof password !== 'string' || password.length < 6) {
      res.status(400).json({ error: 'Password must be at least 6 characters.' });
      return;
    }

    const normalizedEmail = email.trim().toLowerCase();
    const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(normalizedEmail);
    if (existing) {
      res.status(409).json({ error: 'An account with this email already exists.' });
      return;
    }

    const hash = bcrypt.hashSync(password, 10);
    const info = db
      .prepare('INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)')
      .run(fullName.trim(), normalizedEmail, hash);

    const userId = Number(info.lastInsertRowid);
    const token = setSessionCookie(req, res, userId, normalizedEmail, true);

    const user = db
      .prepare('SELECT id, full_name, email, role, institution, created_at FROM users WHERE id = ?')
      .get(userId) as any;

    res.status(201).json({
      token,
      user: {
        id: user.id,
        fullName: user.full_name,
        email: user.email,
        role: user.role,
        institution: user.institution,
        createdAt: user.created_at,
      },
    });
  });

  app.post('/api/auth/login', (req, res) => {
    const { email, password, rememberMe } = req.body ?? {};
    if (!email || typeof email !== 'string' || !password || typeof password !== 'string') {
      res.status(400).json({ error: 'Email and password are required.' });
      return;
    }

    const normalizedEmail = email.trim().toLowerCase();
    const user = db
      .prepare('SELECT * FROM users WHERE email = ?')
      .get(normalizedEmail) as any;

    if (!user || !bcrypt.compareSync(password, user.password_hash)) {
      res.status(401).json({ error: 'Invalid email address or password.' });
      return;
    }

    const token = setSessionCookie(req, res, user.id, user.email, Boolean(rememberMe));
    res.json({
      token,
      user: {
        id: user.id,
        fullName: user.full_name,
        email: user.email,
        role: user.role,
        institution: user.institution,
        createdAt: user.created_at,
      },
    });
  });

  app.post('/api/auth/demo', (req, res) => {
    const user = db
      .prepare('SELECT * FROM users WHERE email = ?')
      .get('researcher@neuropredict.edu') as any;
    if (!user) {
      res.status(404).json({ error: 'Demo researcher account not found.' });
      return;
    }
    const token = setSessionCookie(req, res, user.id, user.email, true);
    res.json({
      token,
      user: {
        id: user.id,
        fullName: user.full_name,
        email: user.email,
        role: user.role,
        institution: user.institution,
        createdAt: user.created_at,
      },
    });
  });

  app.post('/api/auth/logout', (req, res) => {
    const isHttps = isHttpsRequest(req);
    res.clearCookie(SESSION_COOKIE, {
      httpOnly: true,
      sameSite: isHttps ? 'none' : 'lax',
      secure: isHttps,
      partitioned: isHttps,
      path: '/',
    });
    res.json({ ok: true });
  });

  app.get('/api/auth/me', (req, res) => {
    const resolved = resolveValidSession(req);
    if (!resolved) {
      res.json({ user: null });
      return;
    }
    const { session, token } = resolved;
    const user = db
      .prepare('SELECT id, full_name, email, role, institution, created_at FROM users WHERE id = ?')
      .get(session.userId) as any;
    if (!user) {
      res.json({ user: null });
      return;
    }
    res.json({
      token,
      user: {
        id: user.id,
        fullName: user.full_name,
        email: user.email,
        role: user.role,
        institution: user.institution,
        createdAt: user.created_at,
      },
    });
  });

  app.patch('/api/auth/profile', requireAuth, (req: AuthedRequest, res) => {
    const { fullName, role, institution } = req.body ?? {};
    if (!fullName || typeof fullName !== 'string' || fullName.trim().length < 2) {
      res.status(400).json({ error: 'Full name must be at least 2 characters.' });
      return;
    }
    db.prepare('UPDATE users SET full_name = ?, role = ?, institution = ? WHERE id = ?').run(
      fullName.trim(),
      typeof role === 'string' && role.trim() ? role.trim() : req.user!.role,
      typeof institution === 'string' && institution.trim()
        ? institution.trim()
        : req.user!.institution,
      req.user!.id
    );
    const updated = db
      .prepare('SELECT id, full_name, email, role, institution, created_at FROM users WHERE id = ?')
      .get(req.user!.id) as any;
    res.json({
      user: {
        id: updated.id,
        fullName: updated.full_name,
        email: updated.email,
        role: updated.role,
        institution: updated.institution,
        createdAt: updated.created_at,
      },
    });
  });

  // --- Dashboard Stats ---
  app.get('/api/dashboard/stats', requireAuth, (req: AuthedRequest, res) => {
    const rows = db
      .prepare('SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC')
      .all(req.user!.id) as any[];

    const totalPredictions = rows.length;
    const mriScansAnalyzed = rows.filter(
      (r) => r.analysis_type === 'MRI' || r.analysis_type === 'Combined'
    ).length;
    const clinicalRecordsProcessed = rows.filter(
      (r) => r.analysis_type === 'Clinical' || r.analysis_type === 'Combined'
    ).length;

    const categoryMap = new Map<string, number>();
    const timelineMap = new Map<string, { mri: number; clinical: number; combined: number }>();

    for (const r of rows) {
      const cat = r.predicted_category;
      categoryMap.set(cat, (categoryMap.get(cat) ?? 0) + 1);

      const dateKey = String(r.created_at).slice(0, 10);
      const entry = timelineMap.get(dateKey) ?? { mri: 0, clinical: 0, combined: 0 };
      if (r.analysis_type === 'MRI') entry.mri++;
      else if (r.analysis_type === 'Clinical') entry.clinical++;
      else entry.combined++;
      timelineMap.set(dateKey, entry);
    }

    res.json({
      totalPredictions,
      mriScansAnalyzed,
      clinicalRecordsProcessed,
      modelStatusLabel: 'Demo Mode (Weights Pending)',
      categoryDistribution: Array.from(categoryMap.entries()).map(([category, count]) => ({
        category,
        count,
      })),
      activityTimeline: Array.from(timelineMap.entries())
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([date, counts]) => ({ date, ...counts })),
      recentPredictions: rows.slice(0, 5).map(mapPredictionRow),
    });
  });

  // --- Prediction Records ---
  app.get('/api/predictions', requireAuth, (req: AuthedRequest, res) => {
    const rows = db
      .prepare('SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC')
      .all(req.user!.id) as any[];
    res.json({ predictions: rows.map(mapPredictionRow) });
  });

  app.post('/api/predictions', requireAuth, (req: AuthedRequest, res) => {
    const {
      analysisType,
      imageFilename,
      imagePreviewData,
      preprocessedPreviewData,
      mriMetrics,
      clinicalData,
      clinicalPreprocessed,
      predictedCategory,
      modelStatus,
      confidenceNote,
      summaryText,
    } = req.body ?? {};

    if (!['MRI', 'Clinical', 'Combined'].includes(analysisType)) {
      res.status(400).json({ error: 'Invalid analysis type.' });
      return;
    }

    if ((analysisType === 'MRI' || analysisType === 'Combined') && !mriMetrics) {
      res.status(400).json({ error: 'MRI preprocessing metrics are required for MRI analysis.' });
      return;
    }

    if ((analysisType === 'Clinical' || analysisType === 'Combined') && !clinicalData) {
      res.status(400).json({ error: 'Clinical assessment inputs are required.' });
      return;
    }

    const countRow = db.prepare('SELECT COUNT(*) as c FROM predictions').get() as { c: number };
    const seq = String(countRow.c + 1).padStart(4, '0');
    const rand = crypto.randomBytes(2).toString('hex').toUpperCase();
    const id = `NP-2026-${seq}-${rand}`;
    const createdAt = new Date().toISOString();

    db.prepare(
      `INSERT INTO predictions (
        id, user_id, analysis_type, image_filename, image_preview_data,
        preprocessed_preview_data, mri_metrics_json, clinical_data_json,
        clinical_preprocessed_json, predicted_category, model_status,
        confidence_note, summary_text, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    ).run(
      id,
      req.user!.id,
      analysisType,
      imageFilename ?? null,
      imagePreviewData ?? null,
      preprocessedPreviewData ?? null,
      mriMetrics ? JSON.stringify(mriMetrics) : null,
      clinicalData ? JSON.stringify(clinicalData) : null,
      clinicalPreprocessed ? JSON.stringify(clinicalPreprocessed) : null,
      predictedCategory || 'Experimental Demo — Preprocessing Complete (Trained Model Pending)',
      modelStatus || 'Experimental Demo Mode (No Trained ML Weights Loaded)',
      confidenceNote || 'Not Available (Awaiting Validated Trained Model)',
      summaryText || 'Preprocessing and feature extraction completed.',
      createdAt
    );

    const saved = db.prepare('SELECT * FROM predictions WHERE id = ?').get(id);
    res.status(201).json({ prediction: mapPredictionRow(saved) });
  });

  app.delete('/api/predictions/:id', requireAuth, (req: AuthedRequest, res) => {
    const info = db
      .prepare('DELETE FROM predictions WHERE id = ? AND user_id = ?')
      .run(req.params.id, req.user!.id);
    if (info.changes === 0) {
      res.status(404).json({ error: 'Prediction record not found.' });
      return;
    }
    res.json({ ok: true });
  });

  // --- Gemini Explanatory Summary ---
  app.post('/api/predictions/:id/explain', requireAuth, async (req: AuthedRequest, res) => {
    const row = db
      .prepare('SELECT * FROM predictions WHERE id = ? AND user_id = ?')
      .get(req.params.id, req.user!.id) as any;
    if (!row) {
      res.status(404).json({ error: 'Prediction record not found.' });
      return;
    }

    const record = mapPredictionRow(row);
    const prompt = `Explain these brain scan and/or memory check results in simple, everyday English so a common person (non-doctor) can easily understand what stands out and what each number means.
Check Mode: ${record.analysisType}
System Mode: ${record.modelStatus}
${
  record.mriMetrics
    ? `MRI Scan Measurements (${record.imageFilename ?? 'Uploaded Scan'}):
- Left–Right Brain Balance (Symmetry): ${(record.mriMetrics.hemisphericSymmetry * 100).toFixed(1)}%
- One-Sided Spot / Mass Asymmetry: ${typeof record.mriMetrics.asymmetricLesionScore === 'number' ? `${(record.mriMetrics.asymmetricLesionScore * 100).toFixed(0)}%` : 'N/A'}
- Focal Bright Spot Share: ${typeof record.mriMetrics.focalBrightSpotRatio === 'number' ? `${(record.mriMetrics.focalBrightSpotRatio * 100).toFixed(1)}%` : 'N/A'}
- Dark Fluid Space Inside Brain (Ventricle Ratio): ${(record.mriMetrics.darkVentricleRatio * 100).toFixed(1)}%
- Brain Edge & Fold Clarity (Edge Density): ${(record.mriMetrics.edgeDensity * 100).toFixed(1)}%
- Scan Brightness (Mean Intensity): ${(record.mriMetrics.meanIntensity * 100).toFixed(0)}%`
    : ''
}
${
  record.clinicalData
    ? `Memory & Health Check Inputs:
- Age: ${record.clinicalData.age}, Gender: ${record.clinicalData.gender}
- Memory Test Score (MMSE): ${record.clinicalData.mmse}/30
- Daily Thinking Rating (CDR): ${record.clinicalData.cdr}
- Reference Guide: ${record.clinicalPreprocessed?.mmseReferenceBand?.label ?? 'N/A'}`
    : ''
}
Organize the answer into 4 short sections:
1. "What This Scan & Check Shows" (plain words—mention whether one-sided spot/mass asymmetry or widened fluid spaces or memory scores stood out)
2. "Which Brain Conditions & Areas Relate" (simply explain what could relate such as brain tumor/focal mass vs stroke patch vs Alzheimer's/memory thinning, and which brain areas—cortex, hippocampus, ventricles, or one hemisphere—are checked)
3. "Practical Precautions & Questions for Your Doctor"
4. "Important Reminder" (mention this is Demo Mode without trained diagnostic AI weights, so only a doctor or radiologist can diagnose).`;

    try {
      const result = await generateResearchExplanation(prompt);
      db.prepare('UPDATE predictions SET gemini_explanation = ? WHERE id = ?').run(
        result.text,
        record.id
      );
      res.json({ explanation: result.text, modelUsed: result.modelUsed });
    } catch (err: any) {
      const status = err instanceof MissingGeminiKeyError ? err.statusCode : 502;
      res.status(status).json({ error: err?.message || 'Unable to generate Gemini explanation.' });
    }
  });

  app.post('/api/gemini/explain', requireAuth, async (req: AuthedRequest, res) => {
    const { topic, question } = req.body ?? {};
    if (!question || typeof question !== 'string' || question.trim().length < 3) {
      res.status(400).json({ error: 'Please provide a question or topic to explain.' });
      return;
    }

    const prompt = `Topic Context: ${topic || 'NeuroPredict AI Alzheimer’s Research Methodology'}
Question: ${question.trim()}
Provide an accurate, concise academic explanation grounded in OASIS/ADNI literature and standard ML methodology (CNN, Transfer Learning, SVM, Random Forest, Evaluation Metrics).`;

    try {
      const result = await generateResearchExplanation(prompt);
      res.json({ explanation: result.text, modelUsed: result.modelUsed });
    } catch (err: any) {
      const status = err instanceof MissingGeminiKeyError ? err.statusCode : 502;
      res.status(status).json({ error: err?.message || 'Unable to generate Gemini explanation.' });
    }
  });

  // --- Floating AI Research Assistant Chatbot ---
  const handleChatRequest = async (req: AuthedRequest, res: Response) => {
    const { message, history, contextSummary } = req.body ?? {};
    if (!message || typeof message !== 'string' || message.trim().length < 2) {
      res.status(400).json({ error: 'Please enter a question for the AI Research Assistant.' });
      return;
    }

    try {
      const result = await generateChatbotReply({
        message: message.trim(),
        history: Array.isArray(history) ? history : [],
        contextSummary: typeof contextSummary === 'string' ? contextSummary.slice(0, 3500) : undefined,
      });
      res.json({ reply: result.text, modelUsed: result.modelUsed });
    } catch (err: any) {
      const status = err instanceof MissingGeminiKeyError ? err.statusCode : 502;
      res.status(status).json({ error: err?.message || 'Unable to reach Gemini Assistant.' });
    }
  };

  app.post('/api/chat', requireAuth, handleChatRequest);
  app.post('/api/gemini/chat', requireAuth, handleChatRequest);

  // --- Validated PyTorch / MONAI Grad-CAM Bridge ---
  app.get('/api/gradcam/status', requireAuth, (_req, res) => {
    const gradcamUrl = process.env.MONAI_GRADCAM_URL?.trim();
    res.json({
      connected: Boolean(gradcamUrl),
      serviceConfigured: Boolean(gradcamUrl),
      note: gradcamUrl
        ? 'External PyTorch/MONAI Grad-CAM endpoint is configured.'
        : 'No validated PyTorch / MONAI Grad-CAM model service connected. Set MONAI_GRADCAM_URL or load a validated Grad-CAM JSON file exported from python/monai_gradcam_service.py.',
    });
  });

  app.post('/api/gradcam', requireAuth, async (req: AuthedRequest, res) => {
    const gradcamUrl = process.env.MONAI_GRADCAM_URL?.trim();
    if (!gradcamUrl) {
      res.status(503).json({
        status: 'model_not_connected',
        error:
          'No validated PyTorch / MONAI Grad-CAM model service is connected. Genuine brain region highlighting requires a validated model (existing pixel measurements are not lesion detection).',
      });
      return;
    }

    const { imageDataUrl, filename } = req.body ?? {};
    if (!imageDataUrl || typeof imageDataUrl !== 'string') {
      res.status(400).json({ error: 'MRI image data URL is required for Grad-CAM inference.' });
      return;
    }

    try {
      const upstream = await fetch(`${gradcamUrl.replace(/\/$/, '')}/gradcam`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data_url: imageDataUrl, filename: filename || 'mri.png' }),
      });
      const payload = await upstream.json();
      if (!upstream.ok) {
        res.status(upstream.status).json({ error: payload?.error || 'Grad-CAM service returned an error.' });
        return;
      }
      res.json(payload);
    } catch (err: any) {
      res.status(502).json({
        error: err?.message || 'Could not reach the configured PyTorch / MONAI Grad-CAM service.',
      });
    }
  });

  // --- Dataset Explorer Persistence ---
  app.get('/api/datasets', requireAuth, (req: AuthedRequest, res) => {
    const rows = db
      .prepare('SELECT * FROM datasets WHERE user_id = ? ORDER BY created_at DESC')
      .all(req.user!.id) as any[];
    res.json({
      datasets: rows.map((r) => ({
        id: r.id,
        ...JSON.parse(r.summary_json),
        createdAt: r.created_at,
      })),
    });
  });

  app.post('/api/datasets', requireAuth, (req: AuthedRequest, res) => {
    const { summary } = req.body ?? {};
    if (!summary || !summary.name || typeof summary.totalRows !== 'number') {
      res.status(400).json({ error: 'Valid dataset summary is required.' });
      return;
    }

    const info = db
      .prepare(
        'INSERT INTO datasets (user_id, name, source_type, total_rows, missing_count, summary_json) VALUES (?, ?, ?, ?, ?, ?)'
      )
      .run(
        req.user!.id,
        summary.name,
        summary.sourceType || 'Custom Upload',
        summary.totalRows,
        summary.missingCount || 0,
        JSON.stringify(summary)
      );

    res.status(201).json({
      dataset: {
        id: Number(info.lastInsertRowid),
        ...summary,
        createdAt: new Date().toISOString(),
      },
    });
  });

  // Guarantee /api and /api/* never fall through to SPA HTML
  app.all('/api', (_req, res) => {
    res.status(404).json({ error: 'API endpoint not found.' });
  });
  app.all('/api/*', (_req, res) => {
    res.status(404).json({ error: 'API endpoint not found.' });
  });

  app.use((err: any, req: Request, res: Response, next: NextFunction) => {
    if (req.path === '/api' || req.path.startsWith('/api/')) {
      res.setHeader('Content-Type', 'application/json');
      res.status(err?.status || 500).json({
        error: err?.message || 'Unexpected server error while processing API request.',
      });
      return;
    }
    next(err);
  });

  // --- Vite Dev Middleware / Production Static Serving ---
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`NeuroPredict AI server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
