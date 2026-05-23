const sequelize = require('./sequelizeService');
const { DataTypes } = require('sequelize');
require('../models/User');

async function syncDatabase() {
    await sequelize.sync();
    const queryInterface = sequelize.getQueryInterface();
    const table = await queryInterface.describeTable('users');
    if (!table.totpSecret) {
        await queryInterface.addColumn('users', 'totpSecret', {
            type: DataTypes.STRING,
            defaultValue: null,
        });
    }
}

syncDatabase().catch((error) => {
    console.error('Database sync failed', error);
});
