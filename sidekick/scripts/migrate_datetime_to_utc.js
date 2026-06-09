/**
 * Datetime UTC audit script.
 *
 * MongoDB stores all BSON Date values as UTC milliseconds internally.
 * Naive datetimes written by pymongo are stored as UTC already — no data
 * transform is required. This script verifies that all datetime fields in
 * the affected collections are proper BSON Dates (not strings or numbers)
 * and logs any anomalies for manual review.
 *
 * Run from MongoDB Compass shell or terminal:
 *   mongosh "mongodb://admin:admin_password@localhost:27010/sidekick" scripts/migrate_datetime_to_utc.js
 */

const checks = [
  { collection: "strava_activities",  fields: ["created_at", "updated_at", "activity_date"] },
  { collection: "workout_analyses",   fields: ["created_at", "updated_at"] },
  { collection: "coach_memories",     fields: ["created_at", "updated_at", "expires_at"] },
  { collection: "tasks",              fields: ["created_at", "started_at", "completed_at"] },
  { collection: "athletes",           fields: ["created_at", "updated_at"] },
  { collection: "strava_tokens",      fields: ["created_at", "updated_at"] },
  { collection: "daily_analyses",     fields: ["analyzed_at"] },
  { collection: "daily_entries",      fields: ["created_at", "updated_at"] },
  { collection: "training_plans",     fields: ["created_at", "updated_at"] },
  { collection: "prompt_logs",        fields: ["created_at"] },
];

print("\n=== Datetime UTC Audit ===\n");

let totalAnomalies = 0;

for (const { collection, fields } of checks) {
  const coll = db.getCollection(collection);
  const total = coll.countDocuments({});
  print(`Collection: ${collection}  (${total} documents)`);

  for (const field of fields) {
    // Find documents where the field exists but is NOT a BSON Date
    const nonDateFilter = {
      [field]: { $exists: true, $not: { $type: "date" } },
    };
    const anomalyCount = coll.countDocuments(nonDateFilter);
    if (anomalyCount > 0) {
      totalAnomalies += anomalyCount;
      print(`  ⚠  ${field}: ${anomalyCount} document(s) with non-Date type`);
      // Show up to 3 examples
      coll.find(nonDateFilter, { _id: 1, [field]: 1 }).limit(3).forEach(doc => {
        print(`     example _id=${doc._id}  value=${JSON.stringify(doc[field])}`);
      });
    } else {
      print(`  ✓  ${field}: all values are BSON Date`);
    }
  }
}

print(`\n=== Audit complete — ${totalAnomalies} anomaly/anomalies found ===\n`);
