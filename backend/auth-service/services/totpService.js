const QRCode = require('qrcode');
const speakeasy = require('speakeasy');
const CustomError = require('../config/CustomError');

function generateSecret() {
    return speakeasy.generateSecret({ length: 20 }).base32;
}

function verifyCode(secret, code) {
    const token = String(code || '').trim();
    if (!/^\d{6}$/.test(token)) {
        throw new CustomError('Invalid authenticator code', 400);
    }
    const valid = speakeasy.totp.verify({
        secret,
        encoding: 'base32',
        token,
        window: 1,
    });
    if (!valid) {
        throw new CustomError('Invalid authenticator code', 400);
    }
    return true;
}

async function buildSetup(email, secret) {
    const issuer = process.env.APP_NAME || 'SaulAI';
    const otpauthUrl = speakeasy.otpauthURL({
        secret,
        label: `${issuer}:${email}`,
        issuer,
        encoding: 'base32',
    });
    const qrCodeDataUrl = await QRCode.toDataURL(otpauthUrl);
    return {
        issuer,
        otpauthUrl,
        qrCodeDataUrl,
    };
}

module.exports = {
    buildSetup,
    generateSecret,
    verifyCode,
};
