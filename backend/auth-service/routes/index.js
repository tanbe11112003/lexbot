var express = require('express');
const {
    registerUser,
    verifyRegisterTotp,
    validateTokenWithRole,
    loginUser,
    verifyLoginTotp,
    changePassword,
    deleteAccount,
    requestRefreshToken,
} = require('../controllers/AuthController');
var router = express.Router();

router.post('/register', registerUser);
router.post('/register/verify-totp', verifyRegisterTotp);
router.post('/validate', validateTokenWithRole);
router.post('/login', loginUser);
router.post('/login/verify-totp', verifyLoginTotp);
router.post('/change-password', changePassword);
router.delete('/account', deleteAccount);
router.post('/refresh-token', requestRefreshToken);
module.exports = router;
