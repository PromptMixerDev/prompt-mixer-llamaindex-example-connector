export const config = {
	connectorName: 'Llamaindex Connector',
	models: [
		'gpt-4o',
		'gpt-4o-2024-05-13',
		'gpt-4-turbo',
		'gpt-4-turbo-2024-04-09',
		'gpt-4-0125-preview',
		'gpt-4-turbo-preview',
		'gpt-4-vision-preview',
		'gpt-4-1106-vision-preview',
		'gpt-4-1106-preview',
		'gpt-4',
		'gpt-4-32k',
		'gpt-3.5-turbo-0125',
		'gpt-3.5-turbo-1106',
		'gpt-3.5-turbo',
		'gpt-3.5-turbo-instruct',
		'gpt-3.5-turbo-16k',
		'gpt-3.5-turbo-0613',
		'gpt-3.5-turbo-16k-0613',
	],
	properties: [
		{
			id: 'prompt',
			name: 'System Prompt',
			value: 'You are a helpful assistant.',
			type: 'string',
		},
		{
			id: 'max_tokens',
			name: 'Max Tokens',
			value: 4096,
			type: 'number',
		},
		{
			id: 'temperature',
			name: 'Temperature',
			value: 0.7,
			type: 'number',
		},
		{
			id: 'top_p',
			name: 'Top P',
			value: 1,
			type: 'number',
		},
		{
			id: 'frequency_penalty',
			name: 'Frequency Penalty',
			value: 0.5,
			type: 'number',
		},
		{
			id: 'presence_penalty',
			name: 'Presence Penalty',
			value: 0.5,
			type: 'number',
		},
		{
			id: 'stop',
			name: 'Stop Sequences',
			value: ['\n'],
			type: 'array',
		},
		{
			id: 'echo',
			name: 'Echo',
			value: false,
			type: 'boolean',
		},
		{
			id: 'best_of',
			name: 'Best Of',
			value: 1,
			type: 'number',
		},
		{
			id: 'logprobs',
			name: 'LogProbs',
			value: false,
			type: 'boolean',
		},
	],
	settings: [
		{
			id: 'API_KEY',
			name: 'API Key',
			value: '',
			type: 'string',
		},
	],
	description:
		'The LLAMAindex Connector enables working with documents through the LLAMAindex API.',
	author: 'Prompt Mixer',
	iconBase64:
		'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTciIHZpZXdCb3g9IjAgMCAxNiAxNyIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIuOTE1IDMuOTAyNDNDMi42MzEzOSAzLjU4ODQ0IDIuODU1OTIgMy4yNjUxNSAzLjAwMzYzIDMuMTQyNzZDMy4zNzIzNSAzLjE2NTI5IDMuNTU0MjEgMy4xMzEyNyAzLjc4NjQ4IDIuODk3NThDMy44MzU0MSAyLjcwMTQ0IDMuNzE3NzUgMi4yMTc0NyAzLjY1MjggMkMzLjkyNzggMiA0LjEzMDI0IDIuMjY3MzYgNC4xOTcwOCAyLjQwMTA1VjJDNi4xODE2IDIuNDk2NTMgNS42ODcwNCA2Ljc3NjE4IDUuOTM0NyA2LjcyNjYyQzYuMTgyMzcgNi42NzcwNiA4LjQ1NTY1IDYuNzU1MjYgOS4xNzE5NiA2LjgxMjU1QzkuOTM1ODYgNi44MTI1NSAxMS4yMDk3IDcuNTcyNjMgMTEuNTMwNSA3LjQ0Mjc3QzExLjg1MTMgNy4zMTI5MSAxMi4xNzM0IDcuMzE4NjQgMTIuMjk0NCA3LjMzNzczQzEzLjA0MyA3LjQzNzA0IDEzLjE5ODMgOC4xOTM5NCAxMy4xODI0IDguNTU5OTdDMTMuMzU4MSA5LjIzOTg0IDEyLjg5MzYgOS41MzcxMiAxMi42Mzk0IDkuNjAwNzhDMTIuNTQ5MSA5Ljg0NTIzIDEyLjYwMTggMTAuNTQ2MiAxMi42Mzk0IDEwLjg2NjFDMTIuODIzOCAxMS4wMjI4IDEyLjc5MTUgMTEuODYzOSAxMi43NTIzIDEyLjI2NDlDMTMuMDgwNCAxMi44NjA3IDEyLjg4OSAxMy4zODI1IDEyLjc1MjMgMTMuNTY4OUMxMi44OTUzIDE0LjI1MzcgMTIuNzM2NiAxNC44MDgzIDEyLjYzOTQgMTVIMTEuOTM0QzExLjk0OSAxNC44MjA1IDEyLjEyODQgMTQuNzQ4NyAxMi4yMTYyIDE0LjczNTNDMTIuMzM2NiAxNC4yMzExIDEyLjI2NjMgMTMuNTU4IDEyLjIxNjIgMTMuMjg0NEMxMS44NyAxMy4xMzIgMTEuNTgyOCAxMi43MDgzIDExLjQ4MjUgMTIuNTE1NEMxMS41MDUgMTMuNDkzNyAxMS4wMTUzIDE0LjEyMDcgMTAuNzY3NiAxNC4zMTJWMTQuODU0NEgxMC4wODA5QzEwLjA3MzQgMTQuNTY1OSAxMC4yNTM0IDE0LjQ1OTkgMTAuMzQ0MyAxNC40NDI5QzExLjE1NyAxMy40NjQgMTAuNTYzOCAxMi40MTc3IDEwLjE2NTYgMTIuMDE2OUMxMC4wMDAxIDExLjkxOTEgOS45OTYyOSAxMS40OTMzIDEwLjAxNTEgMTEuMjkyNkM5LjE0MjIxIDExLjc2NjcgNy43NTc2MSAxMS42MDkzIDcuMTc0NDMgMTEuNDcxM0M3LjE4OTQ4IDEyLjM0NDIgNy4xNzQ0MyAxMy4wNzk4IDYuODg0MyAxMy4yODQ0VjE0LjQ5OTlDNi44ODQzIDE0LjY5MjMgNi43MDI4MiAxNC45MTM1IDYuNjEyMDggMTVINS45MzQ3QzUuODg5MTIgMTQuNzcyMSA2LjE4MTYgMTQuNTMzNiA2LjMzMzUzIDE0LjQ0MjlDNi4zOTQzMSAxMy42MzI2IDYuMjU3NTcgMTMuMTUxNSA2LjE4MTYgMTMuMDEyMkM2LjE0NjE1IDEzLjcyMTIgNS42OTgzNiAxNC41MzU4IDUuNDc4OSAxNC44NTQ0SDQuOTE1NDhDNC44OTAxNSAxNC41NTU2IDUuMDk5MDYgMTQuNDU1NiA1LjIwNjY4IDE0LjQ0MjlMNS4zNTIyOSAxNC4wODg0QzUuNTI5NTUgMTMuNTc1NiA1LjUxNjg4IDEyLjc5NjkgNS40MTU1OSAxMi4zNzI4QzUuMzM0NTYgMTIuMDMzNSA1LjMzOTYzIDExLjIyNyA1LjM1MjI5IDEwLjg2NjFDMy43NzU5NyA5Ljk1NDUgMy44OTA0IDkuMjQyMSAzLjkwMjU4IDguMjM4OUMzLjU4MzUxIDguMTkzMzIgMy41NzEyOCA3LjA2MzUyIDMuNjA1MDQgNi41MDQzMkMzLjI2OTUyIDUuNTk5MDQgMy42NTU2OCA0LjY5Mzc2IDMuNzI1MzIgNC41NzM0OEMzLjc4MTAzIDQuNDc3MjUgMy43NDg1MyA0LjMwOTcgMy43MjUzMiA0LjIzNzk2QzMuNTczMzkgNC4yNTY5NSAzLjE5ODYxIDQuMjE2NDMgMi45MTUgMy45MDI0M1oiIGZpbGw9IiM4NjhBOTEiLz4KPC9zdmc+Cg==',
};
