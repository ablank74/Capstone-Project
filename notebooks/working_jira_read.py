def validate_and_suggest_it_group_with_confidence(group_name, confidence_threshold=0.20):
    df_tickets = search_jira_unassigned_tickets_in_group(group_name)
    if df_tickets is not None and not df_tickets.empty:
        # Assuming 'summary' and 'description' fields are used for prediction
        # Fill NaN values with an empty string or some default text
        df_tickets['fields.summary'].fillna('', inplace=True)
        df_tickets['fields.description'].fillna('', inplace=True)

        combined_fields = df_tickets['fields.summary'] + ' ' + df_tickets['fields.description']

        # Get Predicted Probabilities
        predicted_probs = pipeline_it_group.predict_proba(combined_fields)
        predicted_groups_encoded = pipeline_it_group.predict(combined_fields)
        predicted_groups = le_it_group.inverse_transform(predicted_groups_encoded)

        # Extract the highest probability as the confidence score
        confidence_scores = predicted_probs.max(axis=1)

        # Base URL for JIRA tickets
        jira_ticket_base_url = "https://csc.cioxhealth.com/browse/"

        # Compare, Suggest and Include Confidence Score and URL
        suggestions = []
        for index, row in df_tickets.iterrows():
            predicted_group = predicted_groups[index]
            confidence_score = confidence_scores[index]

            # Create a URL to the ticket
            ticket_url = jira_ticket_base_url + row['key']

            # Only suggest reassignment if confidence is greater than the threshold
            if confidence_score > confidence_threshold and predicted_group != group_name:
                suggestions.append({
                    'Ticket': f"[{row['key']}]({ticket_url})",
                    'Current Group': group_name,
                    'Suggested Group': predicted_group,
                    'Confidence Score': confidence_score
                })

        # Sort suggestions by confidence score in descending order
        sorted_suggestions = sorted(suggestions, key=lambda x: x['Confidence Score'], reverse=True)
        
        return sorted_suggestions
    else:
        return "No tickets found or error in retrieval"

# Example usage
group_name = "M365"
suggestions = validate_and_suggest_it_group_with_confidence(group_name)

if isinstance(suggestions, list) and len(suggestions) > 0:
    print("Suggestions for reassignment with confidence scores and links to JIRA, sorted by confidence:")
    for suggestion in suggestions:
        print(f"Ticket {suggestion['Ticket']} should be in '{suggestion['Suggested Group']}' instead of '{suggestion['Current Group']}' with confidence score: {suggestion['Confidence Score']:.2f}")
else:
    print(suggestions)