import {
	OpenAI,
	SimpleDirectoryReader,
	ChatMessage,
	ChatResponse,
	Document,
	VectorStoreIndex,
	QueryEngine,
	RetrieverQueryEngine,
	MessageContentDetail,
} from 'llamaindex';
import fs from 'node:fs';
import utils from 'node:util';
import readline from 'node:readline';

import { config } from './config';

const ENCODING: BufferEncoding = 'utf-8';

interface Completion {
	Content: string | null;
	TokenUsage: number | undefined;
}

interface ConnectorResponse {
	Completions: Completion[];
	ModelType: string;
}

interface RawResponse {
	model?: string;
	usage?: {
		prompt_tokens?: number;
		completion_tokens?: number;
		total_tokens?: number;
	};
}

interface Message extends ChatResponse {
	raw: RawResponse | null;
}

interface ErrorCompletion {
	choices: Array<{
		message: {
			content: string;
		};
	}>;
	error: string;
	model: string;
	usage: undefined;
}

interface DocumentWithIndex {
	text: string;
	indexFromDocument: (QueryEngine & RetrieverQueryEngine) | undefined;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const logger = (logBody: any, description?: string): void => {
	if (description) console.log(`${description}`);

	console.log(
		utils.inspect(logBody, {
			showHidden: false,
			depth: null,
			colors: true,
		}),
	);
};

const mapToResponse = (
	outputs: (Message | ErrorCompletion)[],
	model: string,
): ConnectorResponse => {
	return {
		Completions: outputs.map((output: Message) => {
			if ('error' in output) {
				return {
					Content: null,
					TokenUsage: undefined,
					Error: output.error,
				};
			}

			return {
				Content: output.message.content as string,
				TokenUsage: output.raw?.usage?.prompt_tokens,
			};
		}),
		ModelType: model,
	};
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const mapErrorToCompletion = (error: any, model: string): ErrorCompletion => {
	const errorMessage = error.message || JSON.stringify(error);
	return {
		choices: [],
		error: errorMessage,
		model,
		usage: undefined,
	};
};

async function checkDirectory(path: string): Promise<boolean | undefined> {
	try {
		const stat = await fs.promises.stat(path);
		return stat.isDirectory();
	} catch (error) {
		logger(error);
	}
}

async function workWithFolder(
	directoryPath: string,
): Promise<DocumentWithIndex[] | undefined> {
	try {
		const directoryReader = new SimpleDirectoryReader();
		const documents = await directoryReader.loadData(directoryPath);

		return Promise.all(
			documents.map(async (doc) => {
				const index = await indexFromDocument(doc.text);
				return {
					text: doc.text,
					indexFromDocument: index,
				};
			}),
		);
	} catch (error) {
		logger(error);
	}
}

async function readFileByStreams(url: string): Promise<string> {
	return new Promise((resolve, reject) => {
		let data = '';

		let readableStream = fs.createReadStream(url, {
			encoding: ENCODING,
		});

		readableStream.on('error', (err: any) => {
			reject(`There was an error reading the file: ${err.message}`);
		});

		let lineReader = readline.createInterface({ input: readableStream });

		lineReader.on('line', (line: any) => {
			data += line + '\n';
		});

		lineReader.on('close', () => {
			resolve(data);
		});
	});
}

async function getTextFromFile(pathOrUrl: string): Promise<string | undefined> {
	try {
		return await readFileByStreams(pathOrUrl);
	} catch (error) {
		console.error(`Cannot read from ${pathOrUrl}. Error: ${error.message}`);
	}
}

async function workWithFile(
	pathOrUrl: string,
): Promise<DocumentWithIndex | undefined> {
	try {
		const text = (await getTextFromFile(pathOrUrl)) as string;

		const index = await indexFromDocument(text);

		return {
			text,
			indexFromDocument: index as unknown as QueryEngine & RetrieverQueryEngine,
		};
	} catch (error) {
		logger(error);
	}
}

async function parseDocument(
	path: string,
): Promise<
	Promise<DocumentWithIndex[]> | Promise<DocumentWithIndex> | undefined
> {
	try {
		const isADirectory = await checkDirectory(path);
		if (isADirectory) return await workWithFolder(path);
		else return await workWithFile(path);
	} catch (error) {
		console.log(error);
	}
}

function extractDocumentUrls(prompt: string): string[] {
	const urlRegex =
		/(https?:\/\/[^\s]+|[a-zA-Z]:\\[^:<>"|?\n]*|\/[^:<>"|?\n]*)/g;
	const urls = prompt.match(urlRegex) || [];

	return urls.filter((url) => {
		const extensionIndex = url.lastIndexOf('.');
		if (extensionIndex === -1) {
			// If no extension (which can be the case for a directory URL), return true:
			return true;
		}
		const extension = url.slice(extensionIndex);
		return extension;
	});
}

async function indexFromDocument(
	text: string,
): Promise<(QueryEngine & RetrieverQueryEngine) | undefined> {
	try {
		const document = new Document({ text });
		const indexDB = await VectorStoreIndex.fromDocuments([document]);
		return indexDB.asQueryEngine() as unknown as QueryEngine &
			RetrieverQueryEngine;
	} catch (error) {
		console.log(error);
	}
}

const updateHistory = async (
	messageHistory: ChatMessage[],
	messageContent: MessageContentDetail[],
	document: DocumentWithIndex | undefined,
	userPrompt: string,
) => {
	messageContent.push({
		type: 'text',
		text: document?.text as string,
	});

	messageHistory.push({
		role: 'user',
		content: messageContent,
	});

	const results = await document?.indexFromDocument?.query({
		query: userPrompt,
	});

	messageHistory.push({
		role: 'system',
		content: results?.response as string,
	});
};

async function main(
	model: string,
	prompts: string[],
	properties: Record<string, unknown>,
	settings: Record<string, unknown>,
): Promise<ConnectorResponse | undefined> {
	try {
		const total = prompts.length;
		const { prompt, ...restProperties } = properties;
		const systemPrompt = (prompt ||
			config.properties.find((prop) => prop.id === 'prompt')?.value) as string;
		const messageHistory: ChatMessage[] = [
			{ role: 'system', content: [{ type: 'text', text: systemPrompt }] },
		];
		const outputs: Array<Message | ErrorCompletion> = [];

		const llm = new OpenAI({
			model,
			additionalChatOptions: { ...restProperties },
			apiKey: settings?.['API_KEY'] as string,
		});

		try {
			for (let index = 0; index < total; index++) {
				try {
					const userPrompt = prompts[index];
					const docUrls = extractDocumentUrls(userPrompt);
					const messageContent: ChatMessage['content'] = [
						{ type: 'text', text: userPrompt },
					];

					for await (const docUrl of docUrls) {
						try {
							const documents = await parseDocument(docUrl);
							if (Array.isArray(documents)) {
								documents.forEach(async (doc) => {
									await updateHistory(
										messageHistory,
										messageContent,
										doc,
										userPrompt,
									);
								});
							} else {
								await updateHistory(
									messageHistory,
									messageContent,
									documents,
									userPrompt,
								);
							}
						} catch (error) {
							console.log(error);
						}
					}

					logger(messageHistory);

					const response: Message = await llm.chat({
						messages: messageHistory,
					});

					outputs.push(response);

					const assistantResponse = response.message.content || 'No response.';

					messageHistory.push({
						role: 'assistant',
						content: [{ type: 'text', text: assistantResponse as string }],
					});

					logger(response, `Response to prompt ${index + 1} of ${total}`);
				} catch (error) {
					const completionWithError = mapErrorToCompletion(error, model);
					outputs.push(completionWithError);
				}
			}

			return mapToResponse(outputs, model);
		} catch (error) {}
	} catch (error) {
		console.error('Error in main function:', error);
		throw error;
	}
}

export { main, config };
