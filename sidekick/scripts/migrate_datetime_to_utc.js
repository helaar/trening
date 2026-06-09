/**
 * Datetime string → BSON Date migration.
 *
 * Background: repositories that used model_dump(mode="json") serialized
 * datetime fields as ISO 8601 strings before pymongo could write them as
 * BSON Dates. This script finds those string-valued fields and converts
 * them in-place to proper BSON Date objects.
 *
 * MongoDB stores all BSON Dates as UTC milliseconds internally, so no
 * timezone conversion is needed — parsing the ISO string is sufficient.
 *
 * How to run in MongoDB Compass:
 *   1. Open the Compass shell (MongoSH tab at the bottom).
 *   2. Switch to the correct database first: use sidekick
 *   3. Paste and run this script.
 *
 * How to run from the terminal:
 *   mongosh "mongodb://admin:admin_password@localhost:27010/sidekick" \
 *     sidekick/scripts/migrate_datetime_to_utc.js
 */

const checks = [
  { collection: "tasks",           fields: ["created_at", "started_at", "completed_at"] },
  { collection: "daily_entries",   fields: ["created_at", "updated_at"] },
  { collection: "daily_analyses",  fields: ["analyzed_at"] },
  { collection: "training_plans",  fields: ["created_at", "updated_at"] },
  { collection: "prompt_logs",     fields: ["created_at"] },
  { collection: "strava_activities", fields: ["created_at", "updated_at", "activity_date"] },
  { collection: "workout_analyses",  fields: ["created_at", "updated_at"] },
  { collection: "coach_memories",    fields: ["created_at", "updated_at", "expires_at"] },
  { collection: "athletes",          fields: ["created_at", "updated_at"] },
  { collection: "strava_tokens",     fields: ["created_at", "updated_at"] },
];

print("\n=== Datetime migration: string → BSON Date ===\n");
print(`Database: ${db.getName()}\n`);

let totalConverted = 0;
let totalAlreadyOk = 0;

for (const { collection, fields } of checks) {
  const coll = db.getCollection(collection);
  const total = coll.countDocuments({});
  print(`Collection: ${collection}  (${total} documents)`);

  for (const field of fields) {
    // Count documents where the field is stored as a string
    const stringFilter = { [field]: { $type: "string" } };
    const stringCount = coll.countDocuments(stringFilter);

    if (stringCount > 0) {
      // Convert each string value to a BSON Date using $dateFromString
      const result = coll.updateMany(
        stringFilter,
        [
          {
            $set: {
              [field]: {
                $dateFromString: { dateString: `$${field}` },
              },
            },
          },
        ]
      );
      totalConverted += result.modifiedCount;
      print(`  ✓  ${field}: converted ${result.modifiedCount} string(s) → BSON Date`);
    } else {
      const nonDateCount = coll.countDocuments({
        [field]: { $exists: true, $not: { $type: "date" } },
      });
      if (nonDateCount > 0) {
        print(`  ⚠  ${field}: ${nonDateCount} document(s) with unexpected non-string, non-Date type — manual review needed`);
        coll.find({ [field]: { $exists: true, $not: { $type: "date" } } }, { _id: 1, [field]: 1 }).limit(3).forEach(doc => {
          print(`     example _id=${doc._id}  value=${JSON.stringify(doc[field])}`);
        });
      } else {
        totalAlreadyOk++;
        print(`  ✓  ${field}: already BSON Date`);
      }
    }
  }
}

print(`\n=== Migration complete — ${totalConverted} field(s) converted, ${totalAlreadyOk} already correct ===\n`);
