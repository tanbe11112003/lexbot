const { handleError } = require('../services/CommonUtils');
const CustomError = require('../config/CustomError');
const User = require('../models/User');
const TokenUtil = require('../services/jwtService');
const passwordService = require('../services/passwordService');
const { ROLES } = require('../config/constants');
const clientRedis = require('../services/redisService');
const totpService = require('../services/totpService');

const TOTP_SETUP_TTL_SECONDS = Number(process.env.TOTP_SETUP_TTL_SECONDS || 600);

function publicUser(user) {
    const data = { ...user };
    delete data.password;
    delete data.totpSecret;
    delete data.deletedAt;
    return data;
}

async function issueSession(user) {
    const cleanUser = publicUser(user);
    const accessToken = TokenUtil.generateAccessToken(cleanUser);
    const refreshToken = TokenUtil.generateRefreshToken(cleanUser);
    await clientRedis.v4.set(`refreshToken-${cleanUser.id}`, refreshToken);
    return {
        ...cleanUser,
        accessToken,
        refreshToken,
    };
}

function pendingRegisterKey(email) {
    return `pending-register:${String(email || '').trim().toLowerCase()}`;
}

function loginTotpSetupKey(userId) {
    return `login-totp-setup:${userId}`;
}

function buildMfaToken(payload) {
    return TokenUtil.generateMfaToken({
        ...payload,
        scope: 'totp_pending',
    });
}

function mfaResponse({ email, id, purpose, setupRequired, setup, message }) {
    return {
        data: {
            mfaRequired: true,
            setupRequired: Boolean(setupRequired),
            mfaToken: buildMfaToken({ email, id, purpose }),
            email,
            ...(setup || {}),
        },
        message,
        status: 200,
    };
}

async function authenticatedUser(req) {
    const authorization = req.headers.authorization || '';
    if (!authorization || authorization === 'Bearer') {
        throw new CustomError('Token is missing', 400);
    }
    const tokenUser = TokenUtil.decodeToken(authorization.substring(7));
    const user = await User.findOne({ where: { id: tokenUser.id } }).then((data) => data && data.toJSON());
    if (!user) throw new CustomError('No user found', 404);
    return user;
}

function verifyUserTotp(user, code) {
    if (!user.totpSecret) {
        throw new CustomError('Authenticator is not configured for this account', 400);
    }
    totpService.verifyCode(user.totpSecret, code);
}

class AuthController {
    /**
     *
     * @param {import('express').Request} req
     * @param {import('express').Response} res
     */
    async registerUser(req, res) {
        try {
            const newUserInfo = req.body;
            if (!newUserInfo.email || !newUserInfo.password || !newUserInfo.name) {
                throw new CustomError('Name, email and password are required', 400);
            }
            const existingUser = await User.findOne({
                where: {
                    email: newUserInfo.email,
                },
            });
            if (existingUser) throw new CustomError('Email already existed', 409);

            const email = String(newUserInfo.email).trim().toLowerCase();
            const totpSecret = totpService.generateSecret();
            await clientRedis.v4.setEx(
                pendingRegisterKey(email),
                TOTP_SETUP_TTL_SECONDS,
                JSON.stringify({
                    name: newUserInfo.name,
                    phonenum: newUserInfo.phonenum || null,
                    email,
                    password: passwordService.hash(newUserInfo.password),
                    totpSecret,
                }),
            );
            const setup = await totpService.buildSetup(email, totpSecret);
            return res.status(200).json(mfaResponse({
                email,
                purpose: 'register_totp',
                setupRequired: true,
                setup,
                message: 'Authenticator setup required',
            }));
        } catch (error) {
            return handleError(res, error);
        }
    }

    async verifyRegisterTotp(req, res) {
        try {
            const { mfaToken, code } = req.body;
            if (!mfaToken || !code) throw new CustomError('MFA token and authenticator code are required', 400);
            const payload = TokenUtil.decodeMfaToken(mfaToken);
            if (payload.purpose !== 'register_totp' || payload.scope !== 'totp_pending') {
                throw new CustomError('Invalid MFA Token', 401);
            }
            const email = String(payload.email || '').trim().toLowerCase();
            const pendingRaw = await clientRedis.v4.get(pendingRegisterKey(email));
            if (!pendingRaw) throw new CustomError('Registration session expired', 400);
            const pendingUser = JSON.parse(pendingRaw);
            totpService.verifyCode(pendingUser.totpSecret, code);
            const existingUser = await User.findOne({ where: { email } });
            if (existingUser) throw new CustomError('Email already existed', 409);
            const newUser = await User.create({
                name: pendingUser.name,
                phonenum: pendingUser.phonenum,
                email,
                password: pendingUser.password,
                totpSecret: pendingUser.totpSecret,
            }).then((data) => data.toJSON());
            await clientRedis.v4.del(pendingRegisterKey(email));
            return res.status(200).json({
                data: await issueSession(newUser),
                message: 'Register successfully',
                status: 200,
            });
        } catch (error) {
            return handleError(res, error);
        }
    }
    /**
     *
     * @param {import('express').Request} req
     * @param {import('express').Response} res
     */

    async validateTokenWithRole(req, res) {
        try {
            const { authorization, role } = req.headers;
            if (!authorization || authorization === 'Bearer')
                throw new CustomError('Token is missing', 400);
            const token = authorization.substring(7);
            switch (role) {
                case ROLES.ADMIN:
                    const adminUser = TokenUtil.decodeToken(token);

                    if (adminUser.role !== ROLES.ADMIN)
                        throw new CustomError('Permission denied.', 401);
                    if (adminUser.role === ROLES.USER)
                        throw new CustomError('You are not allowed to do this.', 403);
                    delete adminUser.password;
                    return res.status(200).json({
                        data: adminUser,
                        message: 'Valid Token',
                        status: 200,
                    });
                case ROLES.USER:
                    let user = TokenUtil.decodeToken(token);
                    if (user.role !== ROLES.USER && user.role !== ROLES.ADMIN)
                        throw new CustomError('Permission denied', 401);
                    delete user.password;
                    return res.status(200).json({
                        data: user,
                        message: 'Valid Token',
                        status: 200,
                    });
                default:
                    const unknownUser = TokenUtil.decodeToken(token);
                    return res.status(200).json({
                        data: unknownUser,
                        message: 'Valid Token',
                        status: 200,
                    });
            }
        } catch (error) {
            return handleError(res, error);
        }
    }
    /**
     *
     * @param {import('express').Request} req
     * @param {import('express').Response} res
     */
    async loginUser(req, res) {
        try {
            const { email, password } = req.body;
            let existUser = await User.findOne({
                where: {
                    email,
                },
            });
            if (!existUser) throw new CustomError('No user found with email ' + email, 404);
            existUser = existUser.toJSON();
            const isValidPassword = passwordService.compare(password, existUser.password);
            if (!isValidPassword) throw new CustomError('Wrong password', 401);

            if (!existUser.totpSecret) {
                const totpSecret = totpService.generateSecret();
                await clientRedis.v4.setEx(loginTotpSetupKey(existUser.id), TOTP_SETUP_TTL_SECONDS, totpSecret);
                const setup = await totpService.buildSetup(existUser.email, totpSecret);
                return res.status(200).json(mfaResponse({
                    id: existUser.id,
                    email: existUser.email,
                    purpose: 'login_totp_setup',
                    setupRequired: true,
                    setup,
                    message: 'Authenticator setup required',
                }));
            }

            return res.status(200).json(mfaResponse({
                id: existUser.id,
                email: existUser.email,
                purpose: 'login_totp',
                setupRequired: false,
                message: 'Authenticator code required',
            }));
        } catch (error) {
            return handleError(res, error);
        }
    }

    async verifyLoginTotp(req, res) {
        try {
            const { mfaToken, code } = req.body;
            if (!mfaToken || !code) throw new CustomError('MFA token and authenticator code are required', 400);
            const payload = TokenUtil.decodeMfaToken(mfaToken);
            if (!['login_totp', 'login_totp_setup'].includes(payload.purpose) || payload.scope !== 'totp_pending') {
                throw new CustomError('Invalid MFA Token', 401);
            }
            const user = await User.findOne({ where: { id: payload.id } }).then((data) => data && data.toJSON());
            if (!user) throw new CustomError('No user found', 404);

            let secret = user.totpSecret;
            if (payload.purpose === 'login_totp_setup') {
                secret = await clientRedis.v4.get(loginTotpSetupKey(user.id));
                if (!secret) throw new CustomError('Authenticator setup session expired', 400);
            }

            totpService.verifyCode(secret, code);
            if (!user.totpSecret) {
                await User.update({ totpSecret: secret }, { where: { id: user.id } });
                user.totpSecret = secret;
                await clientRedis.v4.del(loginTotpSetupKey(user.id));
            }
            return res.status(200).json({
                data: await issueSession(user),
                message: 'Login successfully!',
                status: 200,
            });
        } catch (error) {
            return handleError(res, error);
        }
    }

    async changePassword(req, res) {
        try {
            const { currentPassword, newPassword, code } = req.body;
            if (!currentPassword || !newPassword || !code) {
                throw new CustomError('Current password, new password and authenticator code are required', 400);
            }
            if (String(newPassword).length < 6) {
                throw new CustomError('New password must be at least 6 characters', 400);
            }
            const user = await authenticatedUser(req);
            const isValidPassword = passwordService.compare(currentPassword, user.password);
            if (!isValidPassword) throw new CustomError('Wrong password', 401);
            verifyUserTotp(user, code);
            await User.update(
                { password: passwordService.hash(newPassword) },
                { where: { id: user.id } },
            );
            return res.status(200).json({
                data: {},
                message: 'Password changed successfully',
                status: 200,
            });
        } catch (error) {
            return handleError(res, error);
        }
    }

    async deleteAccount(req, res) {
        try {
            const { password, code } = req.body;
            if (!password || !code) {
                throw new CustomError('Password and authenticator code are required', 400);
            }
            const user = await authenticatedUser(req);
            const isValidPassword = passwordService.compare(password, user.password);
            if (!isValidPassword) throw new CustomError('Wrong password', 401);
            verifyUserTotp(user, code);
            await clientRedis.v4.del(`refreshToken-${user.id}`);
            await User.destroy({ where: { id: user.id }, force: true });
            return res.status(200).json({
                data: {},
                message: 'Account deleted successfully',
                status: 200,
            });
        } catch (error) {
            return handleError(res, error);
        }
    }
    /**
     *
     * @param {import('express').Request} req
     * @param {import('express').Response} res
     */
    async requestRefreshToken(req, res) {
        try {
            const { refreshToken } = req.body;
            if (!refreshToken) throw new CustomError('Refresh token is missing', 400);
            const refreshTokenInfo = TokenUtil.decodeRefreshToken(refreshToken);

            const refreshTokenInRedis = await clientRedis.v4.get(
                `refreshToken-${refreshTokenInfo.id}`,
            );
            if (refreshToken === refreshTokenInRedis) {
                const user = await User.findOne({
                    where: {
                        id: refreshTokenInfo.id,
                    },
                }).then((data) => data.toJSON());
                const accessToken = TokenUtil.generateAccessToken(publicUser(user));
                return res.status(200).json({
                    data: {
                        accessToken,
                    },
                    message: 'RefreshToken Succeeded',
                    status: 200,
                });
            }
            return res.status(400).json({
                data: {},
                message: 'Invalid refresh Token',
                status: 400,
            });
        } catch (err) {
            return handleError(res, err);
        }
    }
}
module.exports = new AuthController();
