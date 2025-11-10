// ==============================================================================
// NODE.JS (EXPRESS) API SERVER
//
// This file is ALREADY CORRECT. No changes are needed.
// It serves your new 'index.html' file and
// handles the '/api/predict' requests.
// ==============================================================================

const express = require('express');
const axios = require('axios');
const cors = require('cors');
const path = require('path');

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const FASTAPI_URL = 'http://127.0.0.1:8000/predict/';

app.post('/api/predict', async (req, res) => {
    try {
        const patientVitals = req.body;
        console.log('Request received at /api/predict. Forwarding to Python AI server...');

        const fastApiResponse = await axios.post(FASTAPI_URL, patientVitals);

        const prediction = fastApiResponse.data;
        console.log('Prediction received from Python:', prediction);

        res.json(prediction);

    } catch (error) {
        console.error('Error proxying to FastAPI:', error.message);
        res.status(500).json({ error: 'An error occurred with the prediction service.' });
    }
});

const PORT = 5000;
app.listen(PORT, () => {
    console.log(`MERN backend server running on http://localhost:${PORT}`);
    console.log(`Your frontend is now available at http://localhost:${PORT}`);
    console.log(`Make sure your Python server is running on http://127.0.0.1:8000`);
});